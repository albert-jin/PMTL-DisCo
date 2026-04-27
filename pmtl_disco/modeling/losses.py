from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def rumor_verification_loss(
    class_probs: Tensor,
    labels: Tensor,
    eps: float = 1e-12,
    reduction: str = "mean",
) -> Tensor:
    if class_probs.ndim != 2 or class_probs.size(-1) != 2:
        raise ValueError(
            f"Expected class_probs shape [batch, 2], got {tuple(class_probs.shape)}."
        )

    log_probs = class_probs.clamp_min(eps).log()

    if labels.ndim == 1:
        return F.nll_loss(log_probs, labels.long(), reduction=reduction)

    if labels.ndim == 2 and labels.size(-1) == 2:
        per_sample_loss = -(labels.float() * log_probs).sum(dim=-1)
        if reduction == "none":
            return per_sample_loss
        if reduction == "sum":
            return per_sample_loss.sum()
        if reduction == "mean":
            return per_sample_loss.mean()
        raise ValueError(f"Unsupported reduction: {reduction}")

    raise ValueError(f"Unsupported labels shape: {tuple(labels.shape)}")


def distribution_ranking_loss(
    anchor_repr: Tensor,
    positive_repr: Tensor,
    negative_repr: Tensor,
    gamma: float = 0.1,
    reduction: str = "mean",
) -> Tensor:
    if not (anchor_repr.shape == positive_repr.shape == negative_repr.shape):
        raise ValueError(
            "anchor_repr, positive_repr, and negative_repr must share the same shape. "
            f"Got {tuple(anchor_repr.shape)}, {tuple(positive_repr.shape)}, and {tuple(negative_repr.shape)}."
        )

    pos_sim = F.cosine_similarity(anchor_repr, positive_repr, dim=-1)
    neg_sim = F.cosine_similarity(anchor_repr, negative_repr, dim=-1)
    per_sample_loss = F.relu(gamma + pos_sim - neg_sim)

    if reduction == "none":
        return per_sample_loss
    if reduction == "sum":
        return per_sample_loss.sum()
    if reduction == "mean":
        return per_sample_loss.mean()
    raise ValueError(f"Unsupported reduction: {reduction}")


def joint_multitask_loss(
    loss_rv: Optional[Tensor],
    loss_dr: Optional[Tensor],
    alpha1: float = 1.0,
    alpha2: float = 0.01,
) -> Tensor:
    if loss_rv is None and loss_dr is None:
        raise ValueError("At least one of loss_rv or loss_dr must be provided.")

    total: Optional[Tensor] = None
    if loss_rv is not None:
        total = alpha1 * loss_rv
    if loss_dr is not None:
        dr_term = alpha2 * loss_dr
        total = dr_term if total is None else total + dr_term

    assert total is not None
    return total


class JointLoss(nn.Module):
    def __init__(self, alpha1: float = 1.0, alpha2: float = 0.01) -> None:
        super().__init__()
        self.alpha1 = alpha1
        self.alpha2 = alpha2

    def forward(self, loss_rv: Optional[Tensor], loss_dr: Optional[Tensor]) -> Tensor:
        return joint_multitask_loss(loss_rv=loss_rv, loss_dr=loss_dr, alpha1=self.alpha1, alpha2=self.alpha2)
