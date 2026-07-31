import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin  # used for model hub
from transformers.modeling_utils import get_parameter_dtype

from .aggregator import Aggregator
from ..heads.camera_head import CameraHead
from ..heads.dpt_head import DPTHead
from ..heads.track_head import TrackHead


class StreamVGGT(nn.Module, PyTorchModelHubMixin):
    def __init__(
            self, img_size=518, patch_size=14, embed_dim=1024,
            use_camera_head=False, use_point_head=False, use_depth_head=False, use_track_head=False,
        ):
        super().__init__()
        self.aggregator = Aggregator(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim)
        self.camera_head = CameraHead(dim_in=2 * embed_dim) if use_camera_head else None
        self.point_head = DPTHead(dim_in=2 * embed_dim, output_dim=4, activation="inv_log", conf_activation="expp1") if use_point_head else None
        self.depth_head = DPTHead(dim_in=2 * embed_dim, output_dim=2, activation="exp", conf_activation="expp1") if use_depth_head else None
        self.track_head = TrackHead(dim_in=2 * embed_dim, patch_size=patch_size) if use_track_head else None

    @property
    def dtype(self) -> torch.dtype:
        """
        `torch.dtype`: The dtype of the module (assuming that all the module parameters have the same dtype).
        """
        return get_parameter_dtype(self)

    def forward(
        self,
        images: torch.Tensor,
        query_points: torch.Tensor = None,
        past_key_values=None,
        use_cache=False,
        past_frame_idx=0,
    ):
        # If without batch dimension, add it
        if len(images.shape) == 4:
            images = images.unsqueeze(0)
        if query_points is not None and len(query_points.shape) == 2:
            query_points = query_points.unsqueeze(0)

        aggregated_tokens_list, patch_start_idx = self.aggregator(images)

        predictions = {}

        with torch.amp.autocast('cuda', enabled=False):
            if self.camera_head is not None:
                pose_enc_list = self.camera_head(aggregated_tokens_list)
                predictions["pose_enc"] = pose_enc_list[-1]  # pose encoding of the last iteration

            if self.depth_head is not None:
                depth, depth_conf = self.depth_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["depth"] = depth
                predictions["depth_conf"] = depth_conf

            if self.point_head is not None:
                pts3d, pts3d_conf = self.point_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["world_points"] = pts3d
                predictions["world_points_conf"] = pts3d_conf

            if self.track_head is not None and query_points is not None:
                track_list, vis, conf = self.track_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx, query_points=query_points
                )
                predictions["track"] = track_list[-1]  # track of the last iteration
                predictions["vis"] = vis
                predictions["conf"] = conf

        predictions["images"] = images

        return predictions

    def inference(self, images, query_points: torch.Tensor = None, past_key_values=None):
        # If without batch dimension, add it
        if len(images.shape) == 4:
            images = images.unsqueeze(0)
        if query_points is not None and len(query_points.shape) == 2:
            query_points = query_points.unsqueeze(0)

        past_key_values = [None] * self.aggregator.depth
        past_key_values_camera = [None] * self.camera_head.trunk_depth

        predictions = {"images": images, "outputs": []}

        for i in range(images.shape[1]):
            current_img = images[:, i:i+1, ...]
            aggregator_output = self.aggregator(
                current_img, 
                past_key_values=past_key_values,
                use_cache=True, 
                past_frame_idx=i,
            )

            if isinstance(aggregator_output, tuple) and len(aggregator_output) == 3:
                aggregated_tokens, patch_start_idx, past_key_values = aggregator_output
            else:
                aggregated_tokens, patch_start_idx = aggregator_output

            with torch.amp.autocast('cuda', enabled=False):
                if self.camera_head is not None:
                    pose_enc_list, past_key_values_camera = self.camera_head(aggregated_tokens, past_key_values_camera=past_key_values_camera, use_cache=True)
                    pose_enc = pose_enc_list[-1]
                    pose_enc = pose_enc[:, 0, :]

                if self.depth_head is not None:
                    depth, depth_conf = self.depth_head(
                        aggregated_tokens, images=images, patch_start_idx=patch_start_idx
                    )
                    depth = depth[:, 0] 
                    depth_conf = depth_conf[:, 0]

                if self.point_head is not None:
                    pts3d, pts3d_conf = self.point_head(
                        aggregated_tokens, images=images, patch_start_idx=patch_start_idx
                    )
                    world_points = pts3d[:, 0] 
                    world_points_conf = pts3d_conf[:, 0]

                if self.track_head is not None and query_points is not None:
                    track_list, vis, conf = self.track_head(
                        aggregated_tokens, images=images, patch_start_idx=patch_start_idx, query_points=query_points
                )
                    track = track_list[-1][:, 0]
                    query_points = track
                    vis = vis[:, 0]
                    track_conf = conf[:, 0]

            predictions['outputs'].append({
                'world_points': world_points,
                'world_points_conf': world_points_conf,
                'depth': depth,
                'depth_conf': depth_conf,
                'pose_enc': pose_enc,
                **({'track': track,
                    'vis': vis,
                    'track_conf': track_conf}
                if query_points is not None else {})
            })

        return predictions
