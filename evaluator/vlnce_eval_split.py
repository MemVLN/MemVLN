import fcntl
import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import tqdm
from PIL import Image
from habitat import Env
from habitat_baselines.config.default import get_config as get_habitat_config
from hydra.core.global_hydra import GlobalHydra
from habitat_extensions import measures
from common.misc import dict_to_cuda
from common.vln_prompt import resolve_memory_prompt
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
from model.qwen.factory import (
    apply_chat_template,
    build_processor,
    configure_processor,
    count_multimodal_tokens,
    initialize_memory_tokens,
    resolve_model_family,
)
from model.streamvggt.processing import (
    build_online_vggt_input,
    resolve_dense_vggt_input,
)
from split_eval.merge_results import (
    aggregate_records,
    discover_split_files,
    load_split_records,
    validate_expected_episode_count,
)

metric_keys = ["success", "oracle_success", "spl", "distance_to_goal", "ndtw"]


@EVALUATOR_REGISTRY.register()
class VLNCESplitEvaluator():
    def __init__(self, cfg, dataset_name, model=None):
        self.dataset_name = dataset_name
        self.cfg = cfg
        
        self.device = torch.device("cuda:0")
        self.model = model
        
        self.model_family = resolve_model_family(cfg)
        self.processor = build_processor(
            cfg,
            min_frames=1, max_frames=cfg.data.vln.max_num_frames
        )
        self.use_vggt = 'vggt' in cfg.vision3d.name.lower()
        configure_processor(self.processor, cfg, use_vggt=self.use_vggt)

        self.use_mem = True if cfg.llm.get("mem_module", None) is not None else False
        if self.use_mem:
            initialize_memory_tokens(
                self.processor,
                int(cfg.llm.mem_module.memory_token_num),
                self.model_family,
            )

        self.base_save_dir = Path(cfg.exp_dir) / 'eval_results' / dataset_name
        self.base_save_dir.mkdir(parents=True, exist_ok=True)
        
        self.split = cfg.task.vlnce[dataset_name].args.split
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
            gpu_device_id=0,
            fast_habitat=self.fast_habitat,
        )

        self.obs_hist_sample = cfg.data.vln.obs_hist_sample
        self.max_num_frames = cfg.data.vln.max_num_frames
        self.merge_action = cfg.data.vln.merge_action
        self.max_act_hist = cfg.data.vln.max_act_hist
        self.skip_merge_step_obs = cfg.data.vln.skip_merge_step_obs
        self.dense_vggt_input = (
            resolve_dense_vggt_input(cfg.data.vln) if self.use_vggt else False
        )
        self.prompt_format = cfg.data.vln.prompt_format
        self.memory_prompt = resolve_memory_prompt(cfg.data.vln.memory_prompt)
        
        self.navigation_prompt_v1 = "You are a navigation robot to perform the instruction: {instruction} {memory_prompt}{act_hist_prompt}What is your next action?"
        self.navigation_prompt_v2 = "{memory_prompt}{visual_prompt}You are a navigation robot to perform the instruction: {instruction} {act_hist_prompt}What is your next action?"
        
        self.reset()

    def update_model(self, model):
        self.model = model

    def reset(self):
        self.hist_rgb_list = []
        self.hist_act_list = []
        self.pending_action_list = []
        self.curr_x, self.curr_y, self.curr_yaw = 0.0, 0.0, 0.0
        self.step_id = 0
    
    def extract_result(self, action):
        return parse_navigation_action(action)

    def calc_odom(self, action):
        if action == 1:
            self.curr_x += np.sin(np.deg2rad(self.curr_yaw)) * 0.25
            self.curr_y += np.cos(np.deg2rad(self.curr_yaw)) * 0.25
        elif action == 2:
            self.curr_yaw -= 15
            if self.curr_yaw < -180: self.curr_yaw += 360
        elif action == 3:
            self.curr_yaw += 15
            if self.curr_yaw > 180: self.curr_yaw -= 360
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
        y_line = textY
        words = instuction.split(' ')
        x = 10
        line = ""
        for word in words:
            test_line = line + ' ' + word if line else word
            test_line_size, _ = cv2.getTextSize(test_line, font, 0.5, 2)
            if test_line_size[0] > image.shape[1] - x:
                cv2.putText(new_image, line, (x, y_line), font, 0.5, (0, 0, 0), 2)
                line = word
                y_line += textsize[1]+5
            else:
                line = test_line
        if line: cv2.putText(new_image, line, (x, y_line), font, 0.5, (0, 0, 0), 2)
        y_line += textsize[1] + 10
        cv2.putText(new_image, navigation, (x, y_line), font, 0.5, (0, 0, 0), 2)
        return new_image

    def config_env(self) -> Env:
        return Env(config=self.config)

    def eval_action(self, split_id=0, total_splits=1, epoch=None):
        if epoch is not None:
            self.save_dir = self.base_save_dir / f"epoch_{epoch}"
        else:
            self.save_dir = self.base_save_dir / "latest"
        
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.split_result_path = self.save_dir / f"results_split_{split_id}.json"
        if self.save_video:
            os.makedirs(os.path.join(self.save_dir, "video"), exist_ok=True)

        env = self.config_env()
        scene_episode_dict = {}
        for episode in env.episodes:
            if episode.scene_id not in scene_episode_dict:
                scene_episode_dict[episode.scene_id] = []
            scene_episode_dict[episode.scene_id].append(episode)

        done_episodes = set()
        if self.split_result_path.exists():
            with open(self.split_result_path, 'r') as f:
                for line in f:
                    try:
                        res = json.loads(line)
                        done_episodes.add((str(res["scene_id"]), str(res["episode_id"])))
                    except Exception:
                        continue

        video_episode_keys = select_video_episode_keys(
            scene_episode_dict,
            shard_index=split_id,
            shard_count=total_splits,
            limit=self.save_video,
            seed=self.video_selection_seed,
            excluded_keys=done_episodes,
        )
        video_traces = []

        self.model.eval()
        sorted_scenes = sorted(scene_episode_dict.keys())
        
        print(f"[Split {split_id}/{total_splits}] Processing data...")

        for scene in sorted_scenes:
            episodes = scene_episode_dict[scene]
            scene_id = scene.split('/')[-2]

            my_episodes = episodes[split_id::total_splits]
            if len(my_episodes) == 0:
                continue

            process_bar = tqdm.tqdm(my_episodes, desc=f"Split {split_id} | {scene_id}")
            for episode in process_bar:
                episode_id = episode.episode_id
                if (str(scene_id), str(episode_id)) in done_episodes:
                    continue

                episode_instruction = episode.instruction.instruction_text if 'objectnav' not in self.config_path else episode.object_category
                env.current_episode = episode
                observations = env.reset()

                self.reset()
                capture_video = episode_result_key(episode) in video_episode_keys
                trace_actions = []
                trace_navigation_text = []
                trace_generation_steps = []
                trace_generated_text = []
                visual_token_counts = []
                memory_token_counts = []
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

                        if self.prompt_format == 'v1':
                            input_txt = self.navigation_prompt_v1.format(
                                instruction=episode_instruction,
                                memory_prompt=self.memory_prompt,
                                act_hist_prompt=act_hist_prompt
                            )
                            messages = [
                                {
                                    "role": "user",
                                    "content": [
                                        {
                                            "type": "video",
                                            "video": input_image_list,
                                        },
                                        {
                                            "type": "text",
                                            "text": input_txt
                                        },
                                    ],
                                }
                            ]
                        elif self.prompt_format == 'v2':
                            visual_input = []
                            # split into history and current observation
                            if len(input_image_list) > 1:
                                # currently not supported, as max_num_frames is set to 1 -> len(video) = 1
                                visual_input.extend([
                                    {
                                        "type": "text",
                                        "text": "Historical observation: ",
                                    },
                                    {
                                        "type": "video",
                                        "video": input_image_list[:-1],
                                    },
                                ])

                            visual_input.extend([
                                {
                                    "type": "text",
                                    "text": "Current observation: ",
                                },
                                {
                                    "type": "image",
                                    "image": input_image_list[-1],
                                },
                            ])
                            input_txt = self.navigation_prompt_v2.split('{visual_prompt}')[-1].format(
                                instruction=episode_instruction,
                                act_hist_prompt=act_hist_prompt,
                            )
                            messages = [
                                {
                                    "role": "user",
                                    "content": [
                                        {
                                            "type": "text",
                                            "text": self.memory_prompt,
                                        },
                                        *visual_input,
                                        {
                                            "type": "text",
                                            "text": input_txt,
                                        },
                                    ],
                                }
                            ]
                        else:
                            raise ValueError(f"Invalid navigation prompt format: {self.prompt_format}")

                        data_dict = apply_chat_template(
                            self.processor,
                            messages,
                            self.model_family,
                            tokenize=True,
                            add_generation_prompt=True,
                            return_dict=True,
                            padding=True,
                            return_tensors="pt",
                        )
                        if self.model_family == "qwen2_5_vl":
                            token_counts = count_multimodal_tokens(
                                data_dict["input_ids"], self.processor
                            )
                            visual_token_counts.append(
                                int(token_counts["visual_token_count"][0].item())
                            )
                            memory_token_counts.append(
                                int(token_counts["memory_token_count"][0].item())
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
                
                # Episode Finish
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
                if self.model_family == "qwen2_5_vl":
                    result["visual_token_counts"] = visual_token_counts
                    result["memory_token_counts"] = memory_token_counts

                with open(self.split_result_path, 'a') as f:
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
            self.save_dir / f"video_traces_split_{split_id}.json",
        )
        replay_video_traces(
            self.config,
            video_traces,
            save_dir=self.save_dir,
            add_text=self.addtext,
        )

    def merge_and_record(
        self,
        epoch=None,
        expected_splits=None,
        expected_episodes=None,
    ):
        print(f"Merging results from {self.save_dir} ...")
        split_files = discover_split_files(
            self.save_dir,
            expected_splits=expected_splits,
        )
        records = load_split_records(split_files)
        validate_expected_episode_count(records, expected_episodes)
        avg_results = aggregate_records(records)
        avg_results["total_episodes"] = avg_results.pop("count")
        
        final_path = self.save_dir / 'final_results.json'
        with open(str(final_path), 'w') as f:
            json.dump(avg_results, f, indent=2, allow_nan=False)
        
        print(f"Final Results (Epoch {epoch}): {avg_results}")
        return avg_results
