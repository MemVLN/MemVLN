import fcntl
import json
import math
import os
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import tqdm
from PIL import Image
from habitat import Env
from habitat_baselines.config.default import get_config as get_habitat_config
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
from habitat_extensions import measures
from common.misc import dict_to_cuda
from common.vln_prompt import (
    build_navigation_user_message,
    resolve_memory_prompt,
)
from evaluator.build import EVALUATOR_REGISTRY
from evaluator.vlnce_actions import STOP_ACTION, parse_navigation_action
from evaluator.vlnce_rollout import (
    VideoReplayTrace,
    configure_fast_habitat,
    episode_result_key,
    finalize_lazy_measures,
    replay_video_traces,
    reset_episode_model_state,
    save_video_traces,
    select_video_episode_keys,
)
from model.qwen.processing_qwen3_vl import Qwen3VLProcessor
from model.streamvggt.processing import (
    build_online_vggt_input,
    resolve_dense_vggt_input,
)

metric_keys = ["success", "oracle_success", "spl", "distance_to_goal", "ndtw"]


@EVALUATOR_REGISTRY.register()
class VLNCEEvaluator():
    def __init__(self, cfg, dataset_name, model=None):
        self.dataset_name = dataset_name
        self.cfg = cfg
        self.device = torch.device(f"cuda:{os.environ.get('LOCAL_RANK', 0)}")
        self.model = model
        # self.processor = self.model.processor if self.model is not None else None
        self.processor = Qwen3VLProcessor.from_pretrained(
            cfg.llm.cfg_path, padding_side=getattr(cfg.llm, "padding_side", "left"),
            min_frames=1, max_frames=cfg.data.vln.max_num_frames
        )
        self.processor.video_processor.do_sample_frames = False
        self.processor.image_processor.size = {
            'shortest_edge': cfg.data.visual_size.image.shortest_edge,
            'longest_edge': cfg.data.visual_size.image.longest_edge,
        }
        self.processor.video_processor.size = {
            'shortest_edge': cfg.data.visual_size.video.shortest_edge,
            'longest_edge': cfg.data.visual_size.video.longest_edge,
        }
        self.use_vggt = 'vggt' in cfg.vision3d.name.lower()
        self.processor.fix_video_preprocess(use_vggt=self.use_vggt)

        self.use_mem = True if cfg.llm.get("mem_module", None) is not None else False
        if self.use_mem:
            self.processor.__init_memory__(cfg.llm.mem_module.memory_token_num)

        self.base_save_dir = Path(cfg.exp_dir) / 'eval_results' / dataset_name
        self.base_save_dir.mkdir(parents=True, exist_ok=True)

        self.split = cfg.task.vlnce[dataset_name].args.split
        self.env_num = cfg.task.vlnce[dataset_name].args.env_num
        self.save_video = cfg.task.vlnce[dataset_name].args.save_video
        inference_cfg = cfg.get("inference", {})
        self.fast_habitat = bool(inference_cfg.get("fast_habitat", True))
        self.empty_cache_on_episode_end = bool(
            inference_cfg.get("empty_cache_on_episode_end", False)
        )
        self.video_selection_seed = inference_cfg.get(
            "video_selection_seed", cfg.get("rng_seed", 0)
        )
        self.config_path = cfg.task.vlnce[self.dataset_name].args.habitat_config_path
        if not os.path.isabs(self.config_path):
            configs_dir = Path(__file__).parent.parent / "configs"
            self.config_path = str((configs_dir / self.config_path).resolve())
        
        GlobalHydra.instance().clear()
        self.config = get_habitat_config(self.config_path)
        configure_fast_habitat(
            self.config,
            split=self.split,
            gpu_device_id=int(os.environ.get("LOCAL_RANK", 0)),
            fast_habitat=self.fast_habitat,
        )
        print(f"config = {type(self.config)}")
        print(OmegaConf.to_yaml(self.config))
        
        # model hyper-parameters
        self.obs_hist_sample = cfg.data.vln.obs_hist_sample
        self.max_num_frames = cfg.data.vln.max_num_frames
        self.merge_action = cfg.data.vln.merge_action
        self.max_act_hist = cfg.data.vln.max_act_hist
        self.skip_merge_step_obs = cfg.data.vln.skip_merge_step_obs
        self.dense_vggt_input = (
            resolve_dense_vggt_input(cfg.data.vln) if self.use_vggt else False
        )
        self.memory_prompt = resolve_memory_prompt(cfg.data.vln.memory_prompt)
        self.num_future_steps = cfg.data.vln.num_future_steps
        self.best_result = -np.inf
        self.reset()

    def update_model(self, model):
        """Helper function to update model during eval."""
        self.model = model

    def reset(self):
        self.hist_rgb_list = []
        self.hist_act_list = []
        self.pending_action_list = []
        self.curr_x, self.curr_y, self.curr_yaw = 0.0, 0.0, 0.0
        self.step_id = 0
    
    def extract_result(self, action):
        # id: 0-stop, 1-move forward, 2-turn left, 3-turn right
        return parse_navigation_action(action)

    def calc_odom(self, action):
        if action == 1:
            self.curr_x += np.sin(np.deg2rad(self.curr_yaw)) * 0.25
            self.curr_y += np.cos(np.deg2rad(self.curr_yaw)) * 0.25
        elif action == 2:
            self.curr_yaw -= 15
            if self.curr_yaw < -180:
                self.curr_yaw += 360
        elif action == 3:
            self.curr_yaw += 15
            if self.curr_yaw > 180:
                self.curr_yaw -= 360
        elif action == 0:
            pass
        self.curr_x = round(self.curr_x, 3)
        self.curr_y = round(self.curr_y, 3)
    
    def addtext(self, image, instuction, navigation):
        h, w = image.shape[:2]
        new_height = h + 150
        new_image = np.zeros((new_height, w, 3), np.uint8)
        new_image.fill(255)
        new_image[:h, :w] = image

        font = cv2.FONT_HERSHEY_SIMPLEX
        textsize = cv2.getTextSize(instuction, font, 0.5, 2)[0]
        textY = h + (50 + textsize[1]) // 2

        y_line = textY + 0 * textsize[1]

        words = instuction.split(' ')
        max_width = new_image.shape[1]
        x = 10
        line = ""

        for word in words:

            test_line = line + ' ' + word if line else word
            test_line_size, _ = cv2.getTextSize(test_line, font, 0.5, 2)

            if test_line_size[0] > image.shape[1] - x:
                cv2.putText(new_image, line, (x, y_line ), font, 0.5, (0, 0, 0), 2)
                line = word
                y_line += textsize[1]+5
            else:
                line = test_line

        if line:
            cv2.putText(new_image, line, (x, y_line), font, 0.5, (0, 0, 0), 2)

        y_line = y_line + 1 * textsize[1] + 10
        new_image = cv2.putText(new_image, navigation, (x, y_line), font, 0.5, (0, 0, 0), 2)

        return new_image
    
    def config_env(self) -> Env:
        env = Env(config=self.config)
        return env
    
    def eval_action(self, rank, epoch=None):
        if epoch is not None:
            self.save_dir = self.base_save_dir / f"epoch_{epoch}"
        else:
            self.save_dir = self.base_save_dir / "latest"
        
        self.save_dir.mkdir(parents=True, exist_ok=True)
        if self.save_video:
            os.makedirs(os.path.join(self.save_dir, "video"), exist_ok=True)

        env = self.config_env()
        scene_episode_dict = {}
        for episode in env.episodes:
            if episode.scene_id not in scene_episode_dict:
                scene_episode_dict[episode.scene_id] = []
            scene_episode_dict[episode.scene_id].append(episode)

        done_episodes = set()
        if os.path.exists(os.path.join(self.save_dir, f"results.json")):
            with open(os.path.join(self.save_dir, f"results.json"), 'r') as f:
                for line in f:
                    try:
                        res = json.loads(line)
                        done_episodes.add((str(res["scene_id"]), str(res["episode_id"])))
                    except Exception:
                        continue

        video_episode_keys = select_video_episode_keys(
            scene_episode_dict,
            shard_index=rank,
            shard_count=self.env_num,
            limit=self.save_video,
            seed=self.video_selection_seed,
            excluded_keys=done_episodes,
        )
        video_traces = []

        self.model.eval()
        for scene in sorted(scene_episode_dict.keys()):
            episodes = scene_episode_dict[scene]
            scene_id = scene.split('/')[-2]
            print(f"scene_id = {scene_id}")

            assigned_episodes = episodes[rank::self.env_num]
            process_bar = tqdm.tqdm(assigned_episodes, desc=f"scene {scene_id}")
            for episode in process_bar:
                episode_id = episode.episode_id
                if (str(scene_id), str(episode_id)) in done_episodes:
                    print(f"skip already done episode {episode_id}")
                    continue

                episode_instruction = episode.instruction.instruction_text if 'objectnav' not in self.config_path else episode.object_category
                print("episode start:", episode_instruction)
                env.current_episode = episode
                observations = env.reset()

                self.reset()
                capture_video = episode_result_key(episode) in video_episode_keys
                trace_actions = []
                trace_navigation_text = []
                trace_generation_steps = []
                trace_generated_text = []
                data_dict = input_dict = output_dict = None
                while not env.episode_over:
                    rgb = observations['rgb']
                    img = Image.fromarray(rgb)

                    if len(self.pending_action_list) == 0:
                        if self.max_act_hist > 0:
                            raise NotImplementedError
                        else:
                            act_hist_prompt = ""

                        self.hist_rgb_list.append(img)
                        if self.obs_hist_sample == "nearest":
                            obs_hist_sample = list(range(max(0, len(self.hist_rgb_list) - self.max_num_frames), len(self.hist_rgb_list)))
                        elif self.obs_hist_sample == "uniform":
                            obs_hist_sample = np.unique(np.linspace(0, len(self.hist_rgb_list) - 1, self.max_num_frames).astype(int)).tolist()
                        else:
                            raise NotImplementedError(f"obs_hist_sample: {self.obs_hist_sample}")
                        input_image_list = [self.hist_rgb_list[idx] for idx in obs_hist_sample]

                        user_message, input_txt = build_navigation_user_message(
                            self.memory_prompt,
                            input_image_list,
                            episode_instruction,
                            act_hist_prompt,
                        )
                        messages = [user_message]

                        data_dict = self.processor.apply_chat_template(
                            messages,
                            tokenize=True,
                            add_generation_prompt=True,
                            return_dict=True,
                            padding=True,
                            return_tensors="pt",
                        )
                        if self.use_vggt:
                            data_dict["vggt_input"] = build_online_vggt_input(
                                self.processor,
                                messages,
                                data_dict["image_grid_thw"],
                                dense_history=self.dense_vggt_input,
                            )

                        if self.use_mem or self.use_vggt:
                            data_dict.update({"episode_id": [self.dataset_name + '-' + str(episode_id)]})
                            data_dict.update({"clip_id": torch.tensor([self.step_id])})
                            data_dict.update({"current_odom": torch.tensor([[self.curr_x, self.curr_y, self.curr_yaw]])})
                        
                        input_dict = dict_to_cuda(data_dict.copy(), self.device)
                        # t0_generate = time()
                        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                            if hasattr(self.model, "module"):
                                output_dict = self.model.module.generate(input_dict)
                            else:
                                output_dict = self.model.generate(input_dict)

                        actions = output_dict['output_txt'][0]
                        if capture_video:
                            trace_generation_steps.append(self.step_id)
                            trace_generated_text.append(actions)
                        navigation_text = actions
                        parsed_actions = actions.split(", ")
                        if self.merge_action:
                            action = parsed_actions[0]
                            action_index, num = self.extract_result(action)
                            if num is not None and action_index in [1, 2, 3]:
                                repeat_count = int(num / (25 if action_index == 1 else 15))
                                if not 1 <= repeat_count <= 3:
                                    print(f"Warning action: {parsed_actions} -> {action}, repeat_count = {repeat_count}")
                                    repeat_count = max(min(repeat_count, 3), 1)   # clip in secure range
                                self.pending_action_list.extend([action_index] * repeat_count)
                            elif action_index == 0:
                                self.pending_action_list.append(0)
                            if action_index is None or len(self.pending_action_list) == 0:
                                print(
                                    "Unrecognized action; applying deterministic "
                                    f"STOP fail-safe: {parsed_actions} -> {action}"
                                )
                                self.pending_action_list.append(STOP_ACTION)
                        else:
                            for action in parsed_actions:
                                action_index, _ = parse_navigation_action(
                                    action, require_magnitude=False
                                )
                                if action_index is None:
                                    print(
                                        "Unrecognized action; applying deterministic "
                                        f"STOP fail-safe: {parsed_actions} -> {action}"
                                    )
                                    action_index = STOP_ACTION
                                self.pending_action_list.append(action_index)

                        action = self.pending_action_list.pop(0)
                    else:
                        if not self.skip_merge_step_obs:
                            self.hist_rgb_list.append(img)
                        action = self.pending_action_list.pop(0)
                        navigation_text = "Pending action: {}".format(action)

                    if capture_video:
                        trace_actions.append(int(action))
                        trace_navigation_text.append(navigation_text)
                    observations = env.step(action)
                    self.hist_act_list.append(action)
                    self.calc_odom(action)
                    self.step_id += 1

                # end of episide
                finalized_measures = finalize_lazy_measures(env)
                if (
                    "ndtw" in self.config.habitat.task.measurements
                    and "ndtw" not in finalized_measures
                ):
                    raise RuntimeError(
                        "Configured nDTW measure was not finalized at episode end"
                    )
                metrics = env.get_metrics()
                if capture_video:
                    video_traces.append(
                        VideoReplayTrace(
                            scene_id=str(scene_id),
                            episode_id=str(episode_id),
                            instruction=episode_instruction,
                            actions=trace_actions,
                            navigation_text=trace_navigation_text,
                            generation_steps=trace_generation_steps,
                            generated_text=trace_generated_text,
                        )
                    )

                print(f"scene_episode {scene_id}_{episode_id} sucecess: {metrics['success']}, spl: {metrics['spl']}, os: {metrics['oracle_success']}, ne: {metrics['distance_to_goal']}")
                result = {
                    "scene_id": scene_id,
                    "episode_id": episode_id,
                    "episode_instruction": episode_instruction,
                    "steps": self.step_id,
                }
                for key in metric_keys:
                    if key in metrics:
                        result[key] = metrics[key]

                with open(os.path.join(self.save_dir, f"results.json"), 'a') as f:
                    fcntl.flock(f, fcntl.LOCK_EX)
                    try:
                        f.write(json.dumps(result) + '\n')
                        f.flush()
                    finally:
                        fcntl.flock(f, fcntl.LOCK_UN)

                data_dict = input_dict = output_dict = None
                reset_episode_model_state(
                    self.model,
                    self.dataset_name + '-' + str(episode_id),
                    empty_cache_on_episode_end=self.empty_cache_on_episode_end,
                )
        
        env.close()
        save_video_traces(
            video_traces,
            self.save_dir / f"video_traces_rank_{rank}.json",
        )
        replay_video_traces(
            self.config,
            video_traces,
            save_dir=self.save_dir,
            add_text=self.addtext,
        )
        return

    def record(self, is_main_process):
        if not is_main_process:
            return False, {}

        full_results = defaultdict(list)
        result_path = self.save_dir / "results.json"

        if result_path.exists():
            with open(result_path, 'r') as f:
                for line in f:
                    try:
                        res = json.loads(line)
                        for k in metric_keys:
                            if k in res and (not math.isinf(res[k])) and (not math.isnan(res[k])):
                                full_results[k].append(res[k])
                    except json.JSONDecodeError:
                        continue
        
        avg_results = {}
        for k, v in full_results.items():
            avg_results[k] = np.mean(v) if len(v) > 0 else 0.0
        
        avg_results["total_episodes"] = len(full_results["success"])

        target_metric = avg_results['success']
        if target_metric > self.best_result:
            is_best = True
            self.best_result = target_metric
        else:
            is_best = False

        with open(str(self.save_dir / 'final_results.json'), 'w') as f:
            json.dump(avg_results, f, indent=2)
        
        print(f"Final Results on {self.dataset_name}: {avg_results}")

        return is_best, avg_results
