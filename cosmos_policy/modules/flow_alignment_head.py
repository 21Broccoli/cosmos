# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Flow-alignment utilities for injecting geometric priors into the diffusion backbone."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


@dataclass
class FlowAlignmentState:
    """Intermediate tensors emitted by :class:`FlowAlignmentHead` for auxiliary losses/logging."""

    geom_params_B_H_W_C: torch.Tensor
    resized_flow_B_C_H_W: Optional[torch.Tensor]
    resized_mask_B_1_H_W: Optional[torch.Tensor]


class FlowAlignmentHead(nn.Module):
    """Builds flow-aware biases from video latents and optional action-flow contexts."""

    def __init__(
        self,
        hidden_dim: int,
        geom_channels: int,
        flow_context_channels: int = 0,
        temporal_kernel: int = 3,
    ) -> None:
        super().__init__()
        if geom_channels <= 0:
            raise ValueError("geom_channels must be > 0 for FlowAlignmentHead")

        temporal_kernel = max(1, temporal_kernel)
        padding_t = (temporal_kernel - 1) // 2
        mid_channels = max(1, hidden_dim // 2)
        self.video_encoder = nn.Sequential(
            nn.Conv3d(hidden_dim, mid_channels, kernel_size=1),
            nn.SiLU(),
            nn.Conv3d(
                mid_channels,
                geom_channels,
                kernel_size=(temporal_kernel, 3, 3),
                padding=(padding_t, 1, 1),
            ),
        )
        self.geom_to_hidden = nn.Conv2d(geom_channels, hidden_dim, kernel_size=1)
        self.flow_to_hidden: Optional[nn.Conv2d]
        if flow_context_channels > 0:
            self.flow_to_hidden = nn.Conv2d(flow_context_channels, hidden_dim, kernel_size=1)
        else:
            self.flow_to_hidden = None
        self.fuse = nn.Sequential(
            nn.SiLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
            nn.SiLU(),
        )
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.out_linear = nn.Linear(hidden_dim, hidden_dim)
        self.gate = nn.Parameter(torch.zeros(1))

    @staticmethod
    def _resize_spatial(tensor: torch.Tensor, target_hw: Tuple[int, int], mode: str) -> torch.Tensor:
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)
        if tensor.shape[-2:] != target_hw:
            align_corners = mode not in {"nearest", "area"}
            tensor = F.interpolate(tensor, size=target_hw, mode=mode, align_corners=align_corners)
        return tensor

    def forward(
        self,
        video_latents_B_T_H_W_D: torch.Tensor,
        action_latent_indices: torch.Tensor,
        flow_context_B_C_H_W: Optional[torch.Tensor] = None,
        flow_mask_B_C_H_W: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, FlowAlignmentState]:
        """Computes a spatial bias for the action tokens.

        Args:
            video_latents_B_T_H_W_D: Transformer latents reshaped to (B, T, H, W, D).
            action_latent_indices: Time indices locating the action slot per sample (B,).
            flow_context_B_C_H_W: Optional OT-aligned action flow tensor.
            flow_mask_B_C_H_W: Optional mask tensor aligned with the flow context.

        Returns:
            bias_B_H_W_D: Learned bias added to the action tokens.
            FlowAlignmentState: Intermediate tensors for auxiliary supervision/logging.
        """

        B, T, H, W, D = video_latents_B_T_H_W_D.shape
        device = video_latents_B_T_H_W_D.device
        action_latent_indices = action_latent_indices.to(device=device, dtype=torch.long)
        action_latent_indices = torch.clamp(action_latent_indices, 0, T - 1)
        batch_idx = torch.arange(B, device=device)

        video_volume = rearrange(video_latents_B_T_H_W_D, "b t h w d -> b d t h w")
        geom_volume = self.video_encoder(video_volume)  # (B, geom_channels, T, H, W)
        geom_volume = rearrange(geom_volume, "b c t h w -> b t h w c")
        geom_action = geom_volume[batch_idx, action_latent_indices]  # (B, H, W, geom_channels)

        geom_feat = self.geom_to_hidden(rearrange(geom_action, "b h w c -> b c h w"))

        resized_flow_proj = None
        stored_flow = None
        if flow_context_B_C_H_W is not None and self.flow_to_hidden is not None:
            resized_flow = self._resize_spatial(flow_context_B_C_H_W, (H, W), mode="bilinear")
            resized_flow_proj = self.flow_to_hidden(resized_flow)
            geom_feat = geom_feat + resized_flow_proj
            stored_flow = resized_flow

        resized_mask = None
        if flow_mask_B_C_H_W is not None:
            resized_mask = self._resize_spatial(flow_mask_B_C_H_W, (H, W), mode="nearest")
            if resized_mask.shape[1] > 1:
                resized_mask = resized_mask.mean(dim=1, keepdim=True)
            geom_feat = geom_feat * torch.clamp(resized_mask, min=0.0, max=1.0)

        fused = self.fuse(geom_feat)
        fused = rearrange(fused, "b c h w -> b h w c")
        fused = self.out_norm(fused)
        bias = self.out_linear(fused)
        bias = torch.tanh(self.gate) * bias

        state = FlowAlignmentState(
            geom_params_B_H_W_C=geom_action,
            resized_flow_B_C_H_W=stored_flow,
            resized_mask_B_1_H_W=resized_mask,
        )
        return bias, state
