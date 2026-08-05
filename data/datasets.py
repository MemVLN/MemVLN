import gzip
import json
import os
import glob
import math
from bisect import bisect_left

import numpy as np
from accelerate.logging import get_logger
from omegaconf import OmegaConf
from torch.utils.data import Dataset
from tqdm import tqdm
from PIL import Image, ImageOps

from common.vln_prompt import resolve_memory_prompt
from model.qwen.factory import (
    build_processor,
    configure_processor,
    initialize_memory_tokens,
    resolve_model_family,
)
from .build import DATASET_REGISTRY
from .data_utils import calc_relative_pos_and_yaw, merge_actions

logger = get_logger(__name__)

class LeoBase(Dataset):
    def __init__(self, cfg, split):
        super().__init__()
        self.num_frames = getattr(cfg.data, 'num_frames', 32)
        self.model_family = resolve_model_family(cfg)
        self.processor = build_processor(
            cfg,
            min_frames=1, max_frames=self.num_frames,
        )
        self.use_vggt = 'vggt' in cfg.vision3d.name.lower()
        configure_processor(self.processor, cfg, use_vggt=self.use_vggt)



class VLNBase(LeoBase):
    navigation_prompt_v1 = "You are a navigation robot to perform the instruction: {instruction} " + \
                           "{memory_prompt}" + \
                           "{act_hist_prompt}What is your next action?"
    navigation_prompt_v2 = "{memory_prompt}{visual_prompt}" + \
                           "You are a navigation robot to perform the instruction: {instruction} " + \
                           "{act_hist_prompt}What is your next action?"
    vlnce_action_list = [
        "stop", 
        "move forward", 
        "turn left", 
        "turn right"
    ]
    def __init__(self, cfg, split):
        super().__init__(cfg, split)
        self.obs_hist_sample = cfg.data.vln.obs_hist_sample
        self.max_num_frames = cfg.data.vln.max_num_frames
        self.merge_action = cfg.data.vln.merge_action
        self.max_act_hist = cfg.data.vln.max_act_hist
        self.memory_prompt = resolve_memory_prompt(cfg.data.vln.memory_prompt)
        self.sample_step = cfg.data.vln.sample_step
        self.num_future_steps = cfg.data.vln.num_future_steps
        self.num_stop_steps = cfg.data.vln.num_stop_steps
        self.skip_merge_step_obs = cfg.data.vln.skip_merge_step_obs
        self.prompt_format = cfg.data.vln.prompt_format
        realworld_cfg = getattr(cfg.data.vln, 'realworld', None)
        realworld_resize_size = getattr(realworld_cfg, 'image_resize_size', None)
        self.realworld_resize_size = None
        if realworld_resize_size:
            assert len(realworld_resize_size) == 2, \
                f"realworld.image_resize_size must be [width, height], got {realworld_resize_size}"
            self.realworld_resize_size = tuple(int(v) for v in realworld_resize_size)
        if self.memory_prompt:
            if cfg.llm.get("mem_module", None) is None:
                raise ValueError(
                    "data.vln.memory_prompt requires llm.mem_module."
                )
            initialize_memory_tokens(
                self.processor,
                int(cfg.llm.mem_module.memory_token_num),
                self.model_family,
            )
        logger.info(f"Loading {self.__class__.__name__} {split}-set annotations")
        self.meta_data, self.num_demos = self.load_anno(cfg.data.vln)
        if split == 'train' and getattr(cfg.data, 'sort_scene', False):
            self.meta_data = sorted(self.meta_data, key=lambda x: (x['episode_id'], x['clip_id']))

        logger.info(f"Finish loading {self.__class__.__name__} {split}-set annotations, collected {self.num_demos} demos and {len(self)} data\n")

    def load_anno(self, data_cfg):
        raise NotImplementedError()

    def __len__(self):
        return len(self.meta_data)

    def _resize_realworld_frame(self, frame):
        if self.realworld_resize_size is None:
            return frame

        if isinstance(frame, Image.Image):
            image = frame
        else:
            image = Image.open(frame)
        image = ImageOps.exif_transpose(image).convert("RGB")
        if image.size != self.realworld_resize_size:
            resample = getattr(Image, "Resampling", Image).BICUBIC
            image = image.resize(self.realworld_resize_size, resample=resample)
        return image

    def _prepare_visual_path(self, video_path, domain):
        if domain != 'RealWorld' or self.realworld_resize_size is None:
            return video_path
        if isinstance(video_path, (list, tuple)):
            return [self._resize_realworld_frame(frame) for frame in video_path]
        return self._resize_realworld_frame(video_path)

    def __getitem__(self, index):
        meta_data = self.meta_data[index]
        domain = meta_data.get('domain', getattr(self, 'domain', None))
        scene_id = meta_data['scene_id']
        episode_id = meta_data['episode_id']
        clip_id = meta_data['clip_id']
        video_path = self._prepare_visual_path(meta_data['video_path'], domain)
        instruction = meta_data['instruction']
        past_actions = meta_data['past_actions']
        pred_actions = meta_data['pred_actions']
        current_odom = meta_data['current_odom']

        if len(past_actions) > 0:
            act_hist_prompt = f"Your previous actions are: {', '.join(past_actions)}. "
        else:
            act_hist_prompt = ""

        if self.prompt_format == 'v1':
            input_txt = self.navigation_prompt_v1.format(
                instruction=instruction,
                memory_prompt=self.memory_prompt,
                act_hist_prompt=act_hist_prompt,
            )
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": video_path,
                        },
                        {
                            "type": "text",
                            "text": input_txt,
                        }
                    ],
                }
            ]
        elif self.prompt_format == 'v2':
            visual_input = []
            # split into history and current observation
            if len(video_path) > 1:
                # currently not supported, as max_num_frames is set to 1 -> len(video) = 1
                visual_input.extend([
                    {
                        "type": "text",
                        "text": "Historical observation: ",
                    },
                    {
                        "type": "video",
                        "video": video_path[:-1],
                    },
                ])

            visual_input.extend([
                {
                    "type": "text",
                    "text": "Current observation: ",
                },
                {
                    "type": "image",
                    "image": video_path[-1],
                },
            ])
            input_txt = self.navigation_prompt_v2.split('{visual_prompt}')[-1].format(
                instruction=instruction,
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

        output_gt = ', '.join(pred_actions)
        if self.split == 'train':
            messages.append({
                "role": "assistant",
                "content": [{"type": "text", "text": output_gt}]
            })

        data_dict = {
            'source': self.__class__.__name__,
            'scene_id': scene_id,
            'episode_id': episode_id,
            'clip_id': clip_id,
            'messages': messages,
            'input_txt': input_txt,
            'output_gt': output_gt,
            'current_odom': current_odom
        }
        optional_keys = ['sample_weight']
        for key in optional_keys:
            if key in meta_data:
                data_dict[key] = meta_data[key]

        return data_dict


@DATASET_REGISTRY.register()
class VLNMP3DR2R(VLNBase):
    def __init__(self, cfg, split):
        self.domain = 'MP3D'
        self.split = split
        super().__init__(cfg, split)

    def load_anno(self, data_cfg):
        meta_data = []
        num_demos = 0
        anno_dir = data_cfg.anno_dir

        anno_path = os.path.join(anno_dir, 'r2r', self.split, f'{self.split}_sub.json')
        if os.path.exists(anno_path):
            with open(anno_path) as f:
                anno_data = json.load(f)
        else:
            anno_path = os.path.join(anno_dir, 'r2r', self.split, f'{self.split}.json.gz')
            with gzip.open(anno_path, 'rt', encoding='utf-8') as f:
                anno_data = json.load(f)['episodes']

        gt_path = os.path.join(anno_dir, 'r2r', self.split, f'{self.split}_gt.json.gz')
        with gzip.open(gt_path, 'rt', encoding='utf-8') as f:
            gt_data = json.load(f)

        aug_stop_steps = getattr(data_cfg, 'aug_stop_steps', 0)
        if aug_stop_steps > 0:
            with open(os.path.join(anno_dir, 'r2r', self.split, 'last_forward_steps.json')) as f:
                aug_stop_steps_dict = json.load(f)
        else:
            aug_stop_steps_dict = None

        for item in tqdm(anno_data):
            ep_id = item['episode_id']
            traj_id = item['trajectory_id']
            scene_id = item['scene_id'].split('/')[-1].removesuffix('.glb')
            instruction = item['instruction'].strip()
            all_actions = gt_data[str(ep_id)]['actions']
            all_actions = [self.vlnce_action_list[action] for action in all_actions]

            all_odoms = calc_relative_pos_and_yaw(all_actions)

            if aug_stop_steps > 0 and str(ep_id) in aug_stop_steps_dict:
                overwrite_steps = min(aug_stop_steps_dict[str(ep_id)], aug_stop_steps)
                all_actions[-1 - overwrite_steps: -1] = ["stop"] * overwrite_steps

            if self.merge_action:
                all_actions, csd_idx_list = merge_actions(all_actions)
                all_odoms = all_odoms[csd_idx_list]

            for ind in range(0, len(all_actions), self.sample_step):
                if ind == len(all_actions) - 1:
                    # separate process for single-STOP action
                    break

                video_path = []
                past_actions = []
                end_ind = min(len(all_actions), ind + self.num_future_steps)
                pred_actions = all_actions[ind: end_ind]
                current_odom = all_odoms[ind]

                if self.merge_action:
                    current_idx = csd_idx_list[ind]
                else:
                    current_idx = ind

                if self.obs_hist_sample == 'nearest':
                    if self.merge_action and self.skip_merge_step_obs:
                        obs_hist_sample = [csd_idx_list[t] for t in range(max(0, ind + 1 - self.max_num_frames), ind + 1)]
                    else:
                        obs_hist_sample = list(range(max(0, current_idx + 1 - self.max_num_frames), current_idx + 1))
                elif self.obs_hist_sample == 'uniform':
                    obs_hist_sample = np.unique(np.linspace(0, current_idx, self.max_num_frames).astype(int)).tolist()
                else:
                    raise NotImplementedError(f"obs_hist_sample: {self.obs_hist_sample}")

                for hist_ind in obs_hist_sample:
                    video_path.append(os.path.join(anno_dir, 'r2r', self.split, 'traj', str(traj_id), f'{hist_ind}.png'))

                for hist_ind in range(max(0, ind-self.max_act_hist), ind):
                    past_actions.append(all_actions[hist_ind])

                meta_data.append({
                    'scene_id': scene_id,
                    'episode_id': f'r2r-{ep_id}',
                    'clip_id': current_idx,
                    'video_path': video_path,
                    'instruction': instruction,
                    'past_actions': past_actions,
                    'pred_actions': pred_actions,
                    'current_odom': current_odom
                })

            # STOP action
            for _ in range(self.num_stop_steps):
                video_path = []

                if self.merge_action:
                    current_idx = csd_idx_list[len(all_actions) - 1]
                else:
                    current_idx = len(all_actions) - 1

                if self.obs_hist_sample == 'nearest':
                    if self.merge_action and self.skip_merge_step_obs:
                        obs_hist_sample = [csd_idx_list[t] for t in range(max(0, len(all_actions) - self.max_num_frames), len(all_actions))]
                    else:
                        obs_hist_sample = list(range(max(0, current_idx + 1 - self.max_num_frames), current_idx + 1))
                elif self.obs_hist_sample == 'uniform':
                    obs_hist_sample = np.unique(np.linspace(0, current_idx, self.max_num_frames).astype(int)).tolist()
                else:
                    raise NotImplementedError(f"obs_hist_sample: {self.obs_hist_sample}")

                for hist_ind in obs_hist_sample:
                    video_path.append(os.path.join(anno_dir, 'r2r', self.split, 'traj', str(traj_id), f'{hist_ind}.png'))

                for hist_ind in range(max(0, ind-self.max_act_hist), ind):
                    past_actions.append(all_actions[hist_ind])

                meta_data.append({
                    'scene_id': scene_id,
                    'episode_id': f'r2r-{ep_id}',
                    'clip_id': current_idx,
                    'video_path': video_path,
                    'instruction': instruction,
                    'past_actions': past_actions,
                    'pred_actions': ['stop'],
                    'current_odom': all_odoms[-1]
                })

            num_demos += 1

        return meta_data, num_demos
    

class VLNMP3DRxRV2(VLNBase):
    def __init__(self, cfg, split):
        self.domain = 'MP3D'
        self.split = split
        super().__init__(cfg, split)
    
    def load_anno(self, data_cfg):
        meta_data = []
        num_demos = 0
        anno_dir = data_cfg.anno_dir

        anno_path = os.path.join(anno_dir, 'rxr', self.split + '_v2', 'annotations.json')
        with open(anno_path, 'r') as f:
            anno_data = json.load(f)

        aug_stop_steps = getattr(data_cfg, 'aug_stop_steps', 0)
        if aug_stop_steps > 0:
            with open(os.path.join(anno_dir, 'rxr', self.split + '_v2', 'last_forward_steps.json')) as f:
                aug_stop_steps_dict = json.load(f)
        else:
            aug_stop_steps_dict = None
        
        for item in tqdm(anno_data):
            ep_id = item['episode_id']
            traj_id = item['trajectory_id']
            scene_id = item['scene_id']
            instruction = item['instruction']
            all_actions = item['actions']
            all_actions = [self.vlnce_action_list[action] for action in all_actions]

            all_odoms = calc_relative_pos_and_yaw(all_actions)

            if aug_stop_steps > 0 and str(ep_id) in aug_stop_steps_dict:
                overwrite_steps = min(aug_stop_steps_dict[str(ep_id)], aug_stop_steps)
                all_actions[-1 - overwrite_steps: -1] = ["stop"] * overwrite_steps

            if self.merge_action:
                all_actions, csd_idx_list = merge_actions(all_actions)
                all_odoms = all_odoms[csd_idx_list]
            
            for ind in range(0, len(all_actions)):
                video_path = []
                past_actions = []
                pred_actions = all_actions[ind: ind+1]
                current_odom = all_odoms[ind]
                
                if self.merge_action:
                    current_idx = csd_idx_list[ind]
                else:
                    current_idx = ind

                if self.obs_hist_sample == 'nearest':
                    if self.merge_action and self.skip_merge_step_obs:
                        obs_hist_sample = [csd_idx_list[t] for t in range(max(0, ind + 1 - self.max_num_frames), ind + 1)]
                    else:
                        obs_hist_sample = list(range(max(0, current_idx + 1 - self.max_num_frames), current_idx + 1))
                elif self.obs_hist_sample == 'uniform':
                    obs_hist_sample = np.unique(np.linspace(0, current_idx, self.max_num_frames).astype(int)).tolist()
                else:
                    raise NotImplementedError(f"obs_hist_sample: {self.obs_hist_sample}")

                for hist_ind in obs_hist_sample:
                    video_path.append(os.path.join(anno_dir, 'rxr', self.split + '_v2', str(traj_id), 'rgb', f'{hist_ind}.jpg'))

                for hist_ind in range(max(0, ind-self.max_act_hist), ind):
                    past_actions.append(all_actions[hist_ind])

                meta_data.append({
                    'scene_id': scene_id,
                    'episode_id': f'rxr-{ep_id}',
                    'clip_id': current_idx,
                    'video_path': video_path,
                    'instruction': instruction,
                    'past_actions': past_actions,
                    'pred_actions': pred_actions,
                    'current_odom': current_odom
                })

            num_demos += 1

        return meta_data, num_demos


class VLNTrajSumBase(VLNBase):
    traj_sum_prompt_v1 = "You are a navigation robot. Given your observation history from the video, " + \
                         "{memory_prompt}" + \
                         "infer and summarize the most likely instruction."
    traj_sum_prompt_v2 = "{memory_prompt}{visual_prompt}" + \
                         "You are a navigation robot. Given the environment memory, infer and summarize the instruction."
    def __init__(self, cfg, split):
        self.use_sub_inst = cfg.data.vln.use_sub_inst
        super().__init__(cfg, split)

    def __getitem__(self, index):
        meta_data = self.meta_data[index]
        domain = meta_data.get('domain', getattr(self, 'domain', None))
        scene_id = meta_data['scene_id']
        episode_id = meta_data['episode_id']
        clip_id = meta_data['clip_id']
        video_path = meta_data['video_path']
        instruction = meta_data['instruction']
        current_odom = meta_data['current_odom']

        if self.prompt_format == 'v1':
            input_txt = self.traj_sum_prompt_v1.format(memory_prompt=self.memory_prompt)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": video_path,
                        },
                        {
                            "type": "text",
                            "text": input_txt,
                        }
                    ],
                }
            ]
        elif self.prompt_format == 'v2':
            visual_input = []
            # split into history and current observation
            if len(video_path) > 1:
                # currently not supported, as max_num_frames is set to 1 -> len(video) = 1
                visual_input.extend([
                    {
                        "type": "text",
                        "text": "Historical observation: ",
                    },
                    {
                        "type": "video",
                        "video": video_path[:-1],
                    },
                ])

            visual_input.extend([
                {
                    "type": "text",
                    "text": "Current observation: ",
                },
                {
                    "type": "image",
                    "image": video_path[-1],
                },
            ])
            input_txt = self.traj_sum_prompt_v2.split('{visual_prompt}')[-1]
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

        output_gt = instruction
        if self.split == 'train':
            messages.append({
                "role": "assistant",
                "content": [{"type": "text", "text": output_gt}]
            })

        data_dict = {
            'source': self.__class__.__name__,
            'scene_id': scene_id,
            'episode_id': episode_id,
            'clip_id': clip_id,
            'messages': messages,
            'input_txt': input_txt,
            'output_gt': output_gt,
            'current_odom': current_odom
        }

        return data_dict


@DATASET_REGISTRY.register()
class VLNTrajSumMP3DR2R(VLNTrajSumBase):
    def __init__(self, cfg, split):
        self.domain = 'MP3D'
        self.split = split
        super().__init__(cfg, split)

    def load_anno(self, data_cfg):
        meta_data = []
        num_demos = 0
        anno_dir = data_cfg.anno_dir

        anno_path = os.path.join(anno_dir, 'r2r', self.split, f'{self.split}_sub.json')
        if os.path.exists(anno_path):
            with open(anno_path) as f:
                anno_data = json.load(f)
        else:
            anno_path = os.path.join(anno_dir, 'r2r', self.split, f'{self.split}.json.gz')
            with gzip.open(anno_path, 'rt', encoding='utf-8') as f:
                anno_data = json.load(f)['episodes']

        gt_path = os.path.join(anno_dir, 'r2r', self.split, f'{self.split}_gt.json.gz')
        with gzip.open(gt_path, 'rt', encoding='utf-8') as f:
            gt_data = json.load(f)

        for item in tqdm(anno_data):
            ep_id = item['episode_id']
            traj_id = item['trajectory_id']
            scene_id = item['scene_id'].split('/')[-1].removesuffix('.glb')
            instruction = item['instruction'].strip()
            all_actions = gt_data[str(ep_id)]['actions']
            all_actions = [self.vlnce_action_list[action] for action in all_actions]
            num_steps = len(all_actions)

            all_odoms = calc_relative_pos_and_yaw(all_actions)
            current_odom = all_odoms[-1]

            if self.merge_action and self.skip_merge_step_obs:
                _, csd_idx_list = merge_actions(all_actions)

            video_path = []

            if self.obs_hist_sample == 'nearest':
                if self.merge_action and self.skip_merge_step_obs:
                    obs_hist_sample = csd_idx_list[-self.max_num_frames:]
                else:
                    obs_hist_sample = list(range(max(0, num_steps - self.max_num_frames), num_steps))
            elif self.obs_hist_sample == 'uniform':
                obs_hist_sample = np.unique(np.linspace(0, num_steps - 1, self.max_num_frames).astype(int)).tolist()
            else:
                raise NotImplementedError(f"obs_hist_sample: {self.obs_hist_sample}")

            for hist_ind in obs_hist_sample:
                video_path.append(os.path.join(anno_dir, 'r2r', self.split, 'traj', str(traj_id), f'{hist_ind}.png'))

            meta_data.append({
                'scene_id': scene_id,
                'episode_id': f'r2r-{ep_id}',
                'clip_id': num_steps - 1,
                'video_path': video_path,
                'instruction': instruction,
                'current_odom': current_odom
            })

            if self.use_sub_inst and 'sub_instructions' in item and len(item['sub_instructions']) == len(item['sub_trajectories']) - 1:
                for i in range(len(item['sub_instructions']) - 1):
                    sub_instruction = item['sub_instructions'][: i + 1]
                    sub_instruction = [inst.strip() for inst in sub_instruction]
                    sub_instruction = ', '.join(sub_instruction)

                    end_waypoint = item['sub_trajectories'][i + 1]
                    end_frame = item['reference_frames'][end_waypoint]
                    sub_video_path = []

                    current_odom = all_odoms[end_frame]

                    if self.obs_hist_sample == 'nearest':
                        if self.merge_action and self.skip_merge_step_obs:
                            t_rightmost = bisect_left(csd_idx_list, end_frame) - 1
                            obs_hist_sample = [csd_idx_list[t] for t in range(max(0, t_rightmost + 2 - self.max_num_frames), t_rightmost + 1)]
                            obs_hist_sample.append(end_frame)
                        else:
                            obs_hist_sample = list(range(max(0, end_frame + 1 - self.max_num_frames), end_frame + 1))
                    elif self.obs_hist_sample == 'uniform':
                        obs_hist_sample = np.unique(np.linspace(0, end_frame, self.max_num_frames).astype(int)).tolist()
                    else:
                        raise NotImplementedError(f"obs_hist_sample: {self.obs_hist_sample}")

                    for hist_ind in obs_hist_sample:
                        sub_video_path.append(os.path.join(anno_dir, 'r2r', self.split, 'traj', str(traj_id), f'{hist_ind}.png'))

                    meta_data.append({
                        'scene_id': scene_id,
                        'episode_id': f'r2r-{ep_id}',
                        'clip_id': end_frame,
                        'video_path': sub_video_path,
                        'instruction': sub_instruction,
                        'current_odom': current_odom
                    })

            num_demos += 1

        return meta_data, num_demos


@DATASET_REGISTRY.register()
class VLNTrajSumMP3DRxRV2(VLNTrajSumBase):
    def __init__(self, cfg, split):
        self.domain = 'MP3D'
        self.split = split
        super().__init__(cfg, split)
    
    def load_anno(self, data_cfg):
        meta_data = []
        num_demos = 0
        anno_dir = data_cfg.anno_dir

        anno_path = os.path.join(anno_dir, 'rxr', self.split + '_v2', 'annotations.json')
        with open(anno_path, 'r') as f:
            anno_data = json.load(f)
        
        for item in tqdm(anno_data):
            ep_id = item['episode_id']
            traj_id = item['trajectory_id']
            scene_id = item['scene_id']
            instruction = item['instruction']
            all_actions = item['actions']
            all_actions = [self.vlnce_action_list[action] for action in all_actions]
            num_steps = len(all_actions)

            all_odoms = calc_relative_pos_and_yaw(all_actions)
            current_odom = all_odoms[-1]

            if self.merge_action and self.skip_merge_step_obs:
                _, csd_idx_list = merge_actions(all_actions)

            video_path = []

            if self.obs_hist_sample == 'nearest':
                if self.merge_action and self.skip_merge_step_obs:
                    obs_hist_sample = csd_idx_list[-self.max_num_frames:]
                else:
                    obs_hist_sample = list(range(max(0, num_steps - self.max_num_frames), num_steps))
            elif self.obs_hist_sample == 'uniform':
                obs_hist_sample = np.unique(np.linspace(0, num_steps - 1, self.max_num_frames).astype(int)).tolist()
            else:
                raise NotImplementedError(f"obs_hist_sample: {self.obs_hist_sample}")

            for hist_ind in obs_hist_sample:
                video_path.append(os.path.join(anno_dir, 'rxr', self.split + '_v2', str(traj_id), 'rgb', f'{hist_ind}.jpg'))

            meta_data.append({
                'scene_id': scene_id,
                'episode_id': f'rxr-{ep_id}',
                'clip_id': num_steps - 1,
                'video_path': video_path,
                'instruction': instruction,
                'current_odom': current_odom
            })

            num_demos += 1

        return meta_data, num_demos


@DATASET_REGISTRY.register()
class VLNRealWorld(VLNBase):
    def __init__(self, cfg, split):
        self.domain = 'RealWorld'
        self.split = split
        super().__init__(cfg, split)

    @staticmethod
    def _angle_delta_rad(angle, ref_angle):
        return float((angle - ref_angle + np.pi) % (2 * np.pi) - np.pi)

    @staticmethod
    def _find_episode_dirs(root_dirs):
        episode_dirs = []
        for root_dir in root_dirs:
            root_dir = os.path.expanduser(str(root_dir))
            if not os.path.exists(root_dir):
                raise FileNotFoundError(f"RealWorld root_dir does not exist: {root_dir}")

            if os.path.exists(os.path.join(root_dir, 'trajectory_log.txt')):
                episode_dirs.append(root_dir)
                continue

            episode_dirs.extend(sorted(glob.glob(os.path.join(root_dir, 'episode_*'))))

        if not episode_dirs:
            raise FileNotFoundError(f"No RealWorld episodes with trajectory_log.txt found under: {root_dirs}")
        for episode_dir in episode_dirs:
            log_path = os.path.join(episode_dir, 'trajectory_log.txt')
            if not os.path.exists(log_path):
                raise FileNotFoundError(f"Missing trajectory_log.txt in RealWorld episode: {episode_dir}")
        return episode_dirs

    @staticmethod
    def _read_log_rows(log_path):
        rows = []
        with open(log_path, 'r', encoding='utf-8') as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    raise ValueError(f"Empty line in RealWorld trajectory log: {log_path}:{line_no}")
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL row in RealWorld trajectory log: {log_path}:{line_no}") from exc
                if not isinstance(row, dict):
                    raise TypeError(f"RealWorld trajectory row must be a JSON object: {log_path}:{line_no}")
                rows.append(row)

        if not rows:
            raise ValueError(f"RealWorld trajectory log has no rows: {log_path}")
        return rows

    def _sample_weight(
        self,
        mode,
        action_match,
        label_actions,
        normalized_return,
        real_cfg,
    ):
        match_weight = float(real_cfg.match_weight)
        mismatch_weight = float(real_cfg.mismatch_weight)
        stop_weight = float(real_cfg.stop_weight)
        min_weight = float(real_cfg.min_weight)
        max_weight = float(real_cfg.max_weight)

        if 0 in label_actions:
            correction_weight = stop_weight
        else:
            correction_weight = match_weight if action_match else mismatch_weight

        reward_weight = 1.0 + 2.0 * float(normalized_return)
        if mode == 'none':
            weight = 1.0
        elif mode == 'reward':
            weight = reward_weight
        elif mode == 'hybrid':
            weight = correction_weight * reward_weight
        else:
            weight = correction_weight
        return float(np.clip(weight, min_weight, max_weight))

    def _obs_hist_indices(self, current_idx, segment_idx=None, segment_start_indices=None):
        if self.obs_hist_sample == 'nearest':
            if segment_start_indices is not None and self.merge_action and self.skip_merge_step_obs:
                return [
                    segment_start_indices[t]
                    for t in range(max(0, segment_idx + 1 - self.max_num_frames), segment_idx + 1)
                ]
            return list(range(max(0, current_idx + 1 - self.max_num_frames), current_idx + 1))
        if self.obs_hist_sample == 'uniform':
            return np.unique(np.linspace(0, current_idx, self.max_num_frames).astype(int)).tolist()
        raise NotImplementedError(f"obs_hist_sample: {self.obs_hist_sample}")

    @staticmethod
    def _segment_reward(dists, start_idx, end_idx, label_actions, stop_dist_threshold):
        reward = dists[start_idx] - dists[end_idx]
        if 0 in label_actions:
            reward += 1.0 if dists[start_idx] <= stop_dist_threshold else -1.0
        return float(reward)

    @staticmethod
    def _normalized_returns(step_rewards):
        returns = np.cumsum(np.array(step_rewards[::-1], dtype=np.float32))[::-1]
        if len(returns) > 1 and float(returns.std()) > 1e-6:
            return (returns - returns.mean()) / (returns.std() + 1e-6)
        return np.zeros_like(returns)

    def load_anno(self, data_cfg):
        real_cfg = data_cfg.realworld
        root_dirs = list(real_cfg.root_dirs)
        if not root_dirs:
            raise ValueError("data.vln.realworld.root_dirs must contain at least one path")

        episode_dirs = self._find_episode_dirs(root_dirs)
        weight_mode = real_cfg.sample_weight_mode
        stop_dist_threshold = float(real_cfg.stop_dist_threshold)

        meta_data = []
        num_demos = 0
        for episode_dir in tqdm(episode_dirs):
            log_path = os.path.join(episode_dir, 'trajectory_log.txt')
            if not os.path.exists(log_path):
                raise FileNotFoundError(f"Missing trajectory_log.txt in RealWorld episode: {episode_dir}")

            rows = self._read_log_rows(log_path)

            episode_name = os.path.basename(os.path.normpath(episode_dir))
            episode_id = f"realworld-{episode_name}"

            image_paths = [os.path.join(episode_dir, row['image_filename']) for row in rows]
            instructions = [row['instruction'].strip() for row in rows]
            for idx, image_path in enumerate(image_paths):
                if not os.path.exists(image_path):
                    raise FileNotFoundError(f"Missing RealWorld image: {episode_dir} row {idx}: {image_path}")
            for idx, instruction in enumerate(instructions):
                if not instruction:
                    raise ValueError(f"RealWorld instruction is empty: {episode_dir} row {idx}")
            positions = [
                np.array([
                    float(row['step_position']['x']),
                    float(row['step_position']['y']),
                    float(row['step_position']['yaw']),
                ], dtype=np.float32)
                for row in rows
            ]
            requested_actions_per_row = [[int(action) for action in row['requested_actions']] for row in rows]
            executed_actions_per_row = [[int(action) for action in row['executed_actions']] for row in rows]
            model_actions_per_row = [[int(action) for action in row['model_action_raw']] for row in rows]
            clip_ids = [int(row['inference_action_slot']) for row in rows]
            executed_action_ids = []
            for idx, actions in enumerate(executed_actions_per_row):
                if len(actions) != 1:
                    raise ValueError(f"RealWorld executed_actions must be atomic: {episode_dir} row {idx}")
                executed_action_ids.append(actions[0])

            if 0 not in requested_actions_per_row[-1]:
                raise ValueError(f"RealWorld episode must end with stop: {episode_dir}")
            if executed_action_ids[-1] != 0:
                raise ValueError(f"RealWorld executed trajectory must end with stop: {episode_dir}")

            final_pose = positions[-1]
            start_pose = positions[0]
            dists = [float(np.linalg.norm(pos[:2] - final_pose[:2])) for pos in positions]

            sample_specs = []
            executed_action_names = [self.vlnce_action_list[action] for action in executed_action_ids]
            if self.merge_action:
                merged_actions, segment_start_indices = merge_actions(executed_action_names)
            else:
                merged_actions = executed_action_names
                segment_start_indices = list(range(len(executed_action_names)))
            for segment_idx, start_idx in enumerate(segment_start_indices):
                next_idx = (
                    segment_start_indices[segment_idx + 1]
                    if segment_idx + 1 < len(segment_start_indices)
                    else start_idx
                )
                label_actions = executed_action_ids[start_idx:next_idx]
                if not label_actions:
                    label_actions = [executed_action_ids[start_idx]]
                model_actions = model_actions_per_row[start_idx]
                # TODO(realworld-bc): `action_match` is only a clean correction signal when
                # `start_idx` is a real model-inference row. For non-inference merged segment
                # starts, `model_action_raw[start_idx]` may be stale or logging-aligned rather
                # than state-aligned. Consider using a neutral correction weight for those
                # segments instead of treating them as model mistakes.
                sample_specs.append({
                    'row_idx': start_idx,
                    'segment_idx': segment_idx,
                    'segment_start_indices': segment_start_indices,
                    'next_idx': next_idx,
                    'label_actions': label_actions,
                    'model_actions': model_actions,
                    'pred_actions': [merged_actions[segment_idx]],
                    'step_reward': self._segment_reward(
                        dists, start_idx, next_idx, label_actions, stop_dist_threshold
                    ),
                    'action_match': label_actions == model_actions,
                })

            if not sample_specs:
                raise ValueError(f"RealWorld episode produced no training samples: {episode_dir}")

            step_rewards = [spec['step_reward'] for spec in sample_specs]
            normalized_returns = self._normalized_returns(step_rewards)

            episode_sample_count = 0
            for spec_idx, spec in enumerate(sample_specs):
                idx = spec['row_idx']
                label_actions = spec['label_actions']
                model_actions = spec['model_actions']
                clip_id = clip_ids[idx]

                current_pose = positions[idx]
                next_pose = positions[spec['next_idx']]
                rel_pose = np.array([
                    current_pose[0] - start_pose[0],
                    current_pose[1] - start_pose[1],
                    math.degrees(self._angle_delta_rad(current_pose[2], start_pose[2])),
                ], dtype=np.float32)

                obs_hist_sample = self._obs_hist_indices(
                    idx, spec['segment_idx'], spec['segment_start_indices']
                )
                video_path = [image_paths[hist_idx] for hist_idx in obs_hist_sample]

                past_actions = []
                if self.max_act_hist > 0:
                    segment_idx = spec['segment_idx']
                    segment_start = max(0, segment_idx - self.max_act_hist)
                    past_actions = [
                        sample_specs[past_idx]['pred_actions'][0]
                        for past_idx in range(segment_start, segment_idx)
                    ]

                meta_data.append({
                    'domain': self.domain,
                    'scene_id': episode_name,
                    'episode_id': episode_id,
                    'clip_id': clip_id,
                    'video_path': video_path,
                    'instruction': instructions[idx],
                    'past_actions': past_actions,
                    'pred_actions': spec['pred_actions'],
                    'current_odom': rel_pose,
                    'sample_weight': self._sample_weight(
                        weight_mode,
                        spec['action_match'],
                        label_actions,
                        normalized_returns[spec_idx],
                        real_cfg,
                    ),
                    'step_position': current_pose,
                    'next_position': next_pose,
                    'final_pose': final_pose,
                    'progress_reward': step_rewards[spec_idx],
                    'action_match': int(spec['action_match']),
                    'requested_actions': requested_actions_per_row[idx],
                    'executed_actions': label_actions,
                    'model_action_raw': model_actions,
                })
                episode_sample_count += 1

            num_demos += 1

            if episode_sample_count == 0:
                raise ValueError(f"RealWorld episode produced no training samples: {episode_dir}")

        if not meta_data:
            raise ValueError("RealWorld dataset produced no samples")

        return meta_data, num_demos


@DATASET_REGISTRY.register()
class LeoMix(Dataset):
    # for train
    mapping = {
        'vln_mp3d_r2r': VLNMP3DR2R,
        'vln_mp3d_rxr_v2': VLNMP3DRxRV2,
        'vln_traj_sum_mp3d_r2r': VLNTrajSumMP3DR2R,
        'vln_traj_sum_mp3d_rxr_v2': VLNTrajSumMP3DRxRV2,
        'vln_realworld': VLNRealWorld,
    }

    def __init__(self, cfg, split):
        self.datasets = []
        self.dataset_keys = []
        self.use_custom_sampler = cfg.task.leomix.LeoMix.use_custom_sampler
        mix_dict = cfg.task.leomix.LeoMix.mix
        ratio = self._normalize_ratios(cfg.task.leomix.LeoMix.ratio)
        self.ratio_unit = getattr(cfg.task.leomix.LeoMix, 'ratio_unit', 'sample')
        logger.info(f"LeoMix about to load:\n{OmegaConf.to_yaml(mix_dict, resolve=True)}\n")
        for task, v0 in mix_dict.items():
            for source in v0:
                dataset_key = f'{task}_{source}'
                self.dataset_keys.append(dataset_key)
                self.datasets.append(self.mapping[dataset_key](cfg, split))

        if isinstance(ratio, (int, float)):
            ratios = [float(ratio)] * len(self.datasets)
        else:
            ratios = [float(r) for r in ratio]
            assert len(ratios) == len(self.datasets), \
                f"LeoMix ratio length {len(ratios)} must match dataset count {len(self.datasets)}"

        self.index_map = []
        self.index_range = [0]
        self.stream_episode_to_indices = None
        self.stream_episode_to_clip_ids = None
        self.stream_episode_to_orig_episode = None
        if self.ratio_unit == 'episode':
            self._build_episode_index_map(ratios)
        elif self.ratio_unit == 'sample':
            self._build_sample_index_map(ratios)
        else:
            raise ValueError(f"Unsupported LeoMix ratio_unit: {self.ratio_unit}")

        logger.info(f"Indices of LeoMix datasets: {self.index_range}")
        logger.info(f"LeoMix ratio_unit={self.ratio_unit}, total samples={len(self.index_map)}")

    @staticmethod
    def _normalize_ratios(ratio):
        if OmegaConf.is_config(ratio):
            return OmegaConf.to_container(ratio, resolve=True)
        return ratio

    @staticmethod
    def _episode_subset(episode_ids, ratio):
        if ratio <= 0 or len(episode_ids) == 0:
            return []
        if ratio >= 1:
            return list(episode_ids)
        num_select = int(len(episode_ids) * ratio)
        if num_select == 0:
            num_select = 1
        if num_select >= len(episode_ids):
            return list(episode_ids)
        selected_indices = np.unique(np.linspace(0, len(episode_ids) - 1, num_select).astype(int)).tolist()
        return [episode_ids[i] for i in selected_indices]

    @staticmethod
    def _group_episodes(dataset):
        episode_to_items = {}
        episode_order = []
        for local_idx, meta_data in enumerate(dataset.meta_data):
            episode_id = str(meta_data['episode_id'])
            if episode_id not in episode_to_items:
                episode_to_items[episode_id] = []
                episode_order.append(episode_id)
            clip_id = meta_data.get('clip_id', local_idx)
            episode_to_items[episode_id].append((clip_id, local_idx))

        for episode_id, items in episode_to_items.items():
            episode_to_items[episode_id] = sorted(items, key=lambda item: item[0])
        return episode_order, episode_to_items

    def _build_sample_index_map(self, ratios):
        self.index_map = []
        self.index_range = [0]
        for dataset_idx, dataset in enumerate(self.datasets):
            dataset_len = len(dataset)
            num_samples = int(dataset_len * ratios[dataset_idx])
            if dataset_len == 0:
                self.index_range.append(len(self.index_map))
                continue
            for local_pos in range(num_samples):
                self.index_map.append({
                    'dataset_idx': dataset_idx,
                    'dataset_key': self.dataset_keys[dataset_idx],
                    'local_idx': local_pos % dataset_len,
                    'stream_episode_id': None,
                    'orig_episode_id': None,
                })
            self.index_range.append(len(self.index_map))

    def _build_episode_index_map(self, ratios):
        self.index_map = []
        self.index_range = [0]
        self.stream_episode_to_indices = {}
        self.stream_episode_to_clip_ids = {}
        self.stream_episode_to_orig_episode = {}

        for dataset_idx, dataset in enumerate(self.datasets):
            ratio = ratios[dataset_idx]
            episode_order, episode_to_items = self._group_episodes(dataset)
            full_repeats = int(ratio) if ratio >= 1 else 0
            fractional_ratio = ratio - full_repeats if ratio >= 1 else ratio

            repeat_plan = []
            for repeat_idx in range(full_repeats):
                repeat_plan.append((repeat_idx, list(episode_order)))
            fractional_subset = self._episode_subset(episode_order, fractional_ratio)
            if fractional_subset:
                repeat_plan.append((full_repeats, fractional_subset))

            selected_streams = 0
            for repeat_idx, selected_episode_ids in repeat_plan:
                for orig_episode_id in selected_episode_ids:
                    stream_episode_id = (
                        f"{self.dataset_keys[dataset_idx]}::"
                        f"{orig_episode_id}::rep{repeat_idx}"
                    )
                    self.stream_episode_to_indices[stream_episode_id] = []
                    self.stream_episode_to_clip_ids[stream_episode_id] = []
                    self.stream_episode_to_orig_episode[stream_episode_id] = orig_episode_id
                    for clip_id, local_idx in episode_to_items[orig_episode_id]:
                        global_idx = len(self.index_map)
                        self.index_map.append({
                            'dataset_idx': dataset_idx,
                            'dataset_key': self.dataset_keys[dataset_idx],
                            'local_idx': local_idx,
                            'stream_episode_id': stream_episode_id,
                            'orig_episode_id': orig_episode_id,
                        })
                        self.stream_episode_to_indices[stream_episode_id].append(global_idx)
                        self.stream_episode_to_clip_ids[stream_episode_id].append(clip_id)
                    selected_streams += 1

            self.index_range.append(len(self.index_map))
            logger.info(
                f"LeoMix episode ratio selected {selected_streams} streams from "
                f"{len(episode_order)} episodes for {self.dataset_keys[dataset_idx]} "
                f"(ratio={ratio})"
            )

    def __len__(self):
        return len(self.index_map)

    @property
    def processor(self):
        return self.datasets[0].processor

    def __getitem__(self, index):
        entry = self.index_map[index]
        data_dict = self.datasets[entry['dataset_idx']][entry['local_idx']]
        if entry['stream_episode_id'] is not None:
            data_dict = data_dict.copy()
            data_dict['orig_episode_id'] = data_dict.get('episode_id', entry['orig_episode_id'])
            data_dict['episode_id'] = entry['stream_episode_id']
            data_dict['dataset_key'] = entry['dataset_key']
        return data_dict
