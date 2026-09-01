import re
import torch

import numpy as np

from typing import Any
from common.vln_prompt import (
    build_navigation_user_message,
    resolve_memory_prompt,
)
from data.data_utils import dict_to_cuda, merge_actions
from model.qwen.processing_qwen3_vl import Qwen3VLProcessor
from model.streamvggt.processing import (
    build_online_vggt_input,
    resolve_dense_vggt_input,
)


class VLNEvaluator:
    def __init__(
        self,
        model: Any = None,
        cfg: Any = None
    ):
        self.cfg = cfg
        if model is None:
            raise ValueError("Real-world inference requires an initialized model.")
        try:
            self.device = next(model.parameters()).device
        except StopIteration as exc:
            raise ValueError("Cannot infer the real-world device from an empty model.") from exc
        self.model = model
        # For Qwen3
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

        self.obs_hist_sample = cfg.data.vln.obs_hist_sample
        self.max_num_frames = cfg.data.vln.max_num_frames
        self.merge_action = cfg.data.vln.merge_action
        self.max_act_hist = cfg.data.vln.max_act_hist
        self.memory_prompt = resolve_memory_prompt(cfg.data.vln.memory_prompt)
        self.sample_step = cfg.data.vln.sample_step
        self.num_future_steps = cfg.data.vln.num_future_steps
        self.dense_vggt_input = (
            resolve_dense_vggt_input(cfg.data.vln) if self.use_vggt else False
        )
        if not self.use_mem or not cfg.data.vln.memory_prompt:
            raise ValueError(
                "MemVLN real-world inference requires the memory model and "
                "data.vln.memory_prompt=true."
            )
        self.instruction = cfg.instruction
        self.reset_memory()

    def reset_memory(self):
        self.his_rgb_list = []
        self.his_act_list = []
        self.pending_act_list = []
        self.curr_x, self.curr_y, self.curr_yaw = 0.0, 0.0, 0.0
        self.step_id = 0
        if hasattr(self.model, "reset_episode_state"):
            cleared_state = self.model.reset_episode_state()
            print(f"Cleared model episode state: {cleared_state}")
    
    def reset_instruction(self, instruction):
        self.instruction = instruction
        print(f"Instruction updated: '{instruction}'")
    
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
    
    def extract_result(self, action):
        # id: 0-stop, 1-move foward, 2-turn left, 3-turn right
        action = action.lower()
        if "stop" in action:
            return 0, None
        elif "forward" in action:
            match = re.search(r'-?\d+', action)
            if match is None:
                return None, None
            match = match.group()
            return 1, float(match)
        elif "left" in action:
            match = re.search(r'-?\d+', action)
            if match is None:
                return None, None
            match = match.group()
            return 2, float(match)
        elif "right" in action:
            match = re.search(r'-?\d+', action)
            if match is None:
                return None, None
            match = match.group()
            return 3, float(match)

        return None, None
    
    @torch.inference_mode()
    def step(self, rgb, episode_id):
        result = self.step_with_metadata(rgb, episode_id)
        return result[0] if isinstance(result, tuple) else result

    @torch.inference_mode()
    def step_with_metadata(self, rgb, episode_id):
        """
        Returns (action, pending_count_after, is_inference, raw_output, action_sequence)
        - action: the atomic action returned to caller
        - pending_count_after: number of pending actions AFTER this step
        - is_inference: True if model inference was triggered, False if served from buffer
        - raw_output: original model output string or list (for buffer pops, the next buffered action)
        - action_sequence: full expanded action sequence when merge_action inference emits one
        """
        if isinstance(rgb, np.ndarray):
            from PIL import Image
            rgb = Image.fromarray(rgb)
        self.his_rgb_list.append(rgb)

        is_inference = False
        raw_output = None

        # fetch pending actions
        if len(self.pending_act_list) > 0:
            action = self.pending_act_list.pop(0)
            pending_count_after = len(self.pending_act_list)  # count AFTER pop
            raw_output = action  # for buffered actions, raw_output is just the action itself
            self.his_act_list.append(action)
            self.calc_odom(action)
            self.step_id += 1
            return action, pending_count_after, is_inference, raw_output, [action]

        if self.max_act_hist > 0:
            act_hist_prompt = f"Your previous actions are: {', '.join(merge_actions(self.his_act_list[-self.max_act_hist:])[0])}."
        else:
            act_hist_prompt = ""
        
        num_steps = len(self.his_rgb_list)
        if self.obs_hist_sample == 'nearest':
            obs_hist_sample = list(range(max(0, num_steps - self.max_num_frames), num_steps))
        elif self.obs_hist_sample == 'uniform':
            obs_hist_sample = np.unique(np.linspace(0, num_steps - 1, self.max_num_frames).astype(int)).tolist()
        else:
            raise NotImplementedError(f"obs_hist_sample: {self.obs_hist_sample}")
        
        input_image_list = [self.his_rgb_list[idx] for idx in obs_hist_sample]
        user_message, input_txt = build_navigation_user_message(
            self.memory_prompt,
            input_image_list,
            self.instruction,
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
            data_dict.update({
                "episode_id": [str(episode_id)],
                "clip_id": torch.tensor([self.step_id]),
                "current_odom": torch.tensor([[self.curr_x, self.curr_y, self.curr_yaw]]),
            })
        # breakpoint()
        input_dict = dict_to_cuda(data_dict.copy(), self.device)
        
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
            output_dict = self.model.generate(
                input_dict,
                num_beams=1,
                num_return_sequences=1,
            )

        actions = output_dict["output_txt"][0]
        
        if self.merge_action:
            action = actions.split(", ")[0]
            action_index, num = self.extract_result(action)
            action_sequence = []
            if num is not None and action_index in [1, 2, 3]:
                repeat_count = int(num / (25 if action_index == 1 else 15))
                repeat_count = max(1, min(repeat_count, 3))
                action_sequence = [action_index] * repeat_count
            elif action_index == 0:
                action_sequence = [0]
            if action_index is None or len(action_sequence) == 0:
                print(f"Unrecognized action, using fail-safe STOP: {actions!r}")
                action_sequence = [0]

            action = action_sequence[0]
            self.pending_act_list = action_sequence[1:]
            pending_count_after = len(self.pending_act_list)
            self.his_act_list.append(action)
            self.calc_odom(action)
            self.step_id += 1
            return action, pending_count_after, True, actions, action_sequence
        else:
            output_actions = actions
            parsed_actions = []
            for action in output_actions.split(", "):
                if "stop" in action:
                    parsed_actions.append(0)
                elif "forward" in action:
                    parsed_actions.append(1)
                elif "left" in action:
                    parsed_actions.append(2)
                elif "right" in action:
                    parsed_actions.append(3)
                else:
                    print(f"Unrecognized action, using fail-safe STOP: {action!r}")
                    parsed_actions.append(0)

            self.pending_act_list = parsed_actions
        
        action = self.pending_act_list.pop(0)
        pending_count_after = len(self.pending_act_list)  # count AFTER pop
        self.his_act_list.append(action)
        self.calc_odom(action)
        self.step_id += 1

        is_inference = True
        return action, pending_count_after, is_inference, output_actions, parsed_actions
