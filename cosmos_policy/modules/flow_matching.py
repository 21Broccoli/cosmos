# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Utility losses for aligning action flows with video flows."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def _reduce_with_mask(tensor: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    if mask is None:
        return tensor.mean()
    weight = mask.to(device=tensor.device, dtype=tensor.dtype)
    if weight.dim() == 3:
        weight = weight.unsqueeze(1)
    if weight.shape[1] == 1 and tensor.shape[1] > 1:
        weight = weight.expand(-1, tensor.shape[1], -1, -1)
    weighted = tensor * weight
    denom = weight.sum().clamp(min=1.0)
    return weighted.sum() / denom


def compute_flow_matching_loss(
    video_flow: torch.Tensor,
    action_flow_pred: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    loss_type: str = "l2",
) -> torch.Tensor:
    """Compute a flow-alignment loss between video-derived flow and predicted action flow."""

    if video_flow.shape != action_flow_pred.shape:
        raise ValueError(
            f"Flow tensors must match in shape: video_flow={video_flow.shape}, action_flow_pred={action_flow_pred.shape}"
        )

    video_flow = video_flow.to(action_flow_pred.dtype)
    diff = action_flow_pred - video_flow

    if loss_type.lower() == "l2":
        per_elem = diff.pow(2)
    elif loss_type.lower() in {"smooth_l1", "huber"}:
        per_elem = F.smooth_l1_loss(action_flow_pred, video_flow, reduction="none")
    elif loss_type.lower() == "cosine":
        eps = 1e-6
        pred_norm = torch.linalg.vector_norm(action_flow_pred, dim=1, keepdim=True).clamp_min(eps)
        video_norm = torch.linalg.vector_norm(video_flow, dim=1, keepdim=True).clamp_min(eps)
        cosine = (action_flow_pred * video_flow).sum(dim=1, keepdim=True) / (pred_norm * video_norm)
        per_elem = (1.0 - cosine).clamp(min=0.0)
    else:
        raise ValueError(f"Unsupported flow matching loss_type={loss_type}")

    return _reduce_with_mask(per_elem, mask)
