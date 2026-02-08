# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Light-weight decoder that maps spatial action-flow latents back to discrete robot actions."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ActionFlowDecoder(nn.Module):
    """
    Maps flow/vector-field latents back to action chunks.

    The decoder is intentionally tiny (two 1x1 convolutions + pooling + linear head)
    so that it can be trained jointly with the diffusion backbone.
    """

    def __init__(self, in_channels: int, hidden_dim: int = 512, dropout: float = 0.0):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.out_proj = None  # Lazily initialized once chunk/action dims are known

    def _build_out_proj(self, chunk_size: int, action_dim: int, device: torch.device):
        out_dim = chunk_size * action_dim
        self.out_proj = nn.Linear(self.hidden_dim, out_dim, device=device)

    def forward(
        self,
        flow: torch.Tensor,
        chunk_size: int,
        action_dim: int,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            flow: (B, C_flow, H, W) action-flow latent channels.
            chunk_size: Number of actions per chunk.
            action_dim: Action dimension (e.g., 14 for ALOHA).
            mask: Optional spatial mask (B, C_mask, H, W) highlighting valid flow pixels.
        """

        if mask is not None:
            if mask.shape[1] == 1:
                mask = mask.expand(-1, self.in_channels, -1, -1)
            flow = flow * mask

        features = self.encoder(flow)
        pooled = F.adaptive_avg_pool2d(features, output_size=1).flatten(1)
        pooled = self.norm(pooled)

        if self.out_proj is None or self.out_proj.out_features != chunk_size * action_dim:
            self._build_out_proj(chunk_size, action_dim, pooled.device)

        action_vec = self.out_proj(pooled)
        return action_vec.view(flow.shape[0], chunk_size, action_dim)
