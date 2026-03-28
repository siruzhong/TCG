"""Temporal-Contextual Gating (TCG): adaptive per-position modulation via local dynamics."""

from __future__ import annotations

from typing import Dict, Optional, Union

import torch
import torch.nn.functional as F
from torch import nn


def tcg_orthogonal_loss(mode_table: torch.Tensor) -> torch.Tensor:
    """
    Encourage rows of mode_table (K x d) to be orthonormal in expectation.

    Args:
        mode_table: Tensor of shape [K, d].

    Returns:
        Scalar: sum of squared Gram errors vs identity, divided by K (not K^2),
        so gradients to ``mode_table`` are not diluted by the full matrix size.
    """
    m = F.normalize(mode_table, dim=-1, eps=1e-6)
    gram = m @ m.T
    k = m.size(0)
    eye = torch.eye(k, device=m.device, dtype=m.dtype)
    return torch.sum((gram - eye) ** 2) / k


class TemporalContextualGating(nn.Module):
    """
    Residual adapter on hidden states [B, L, d].

    Multi-scale context uses depthwise Conv1d (``groups=d_model``): O(d*k) params per
    kernel, per-dimension local dynamics along L; mixing happens in ``context_mlp``.
    Then softmax routing over K learnable patterns and Hadamard modulation x * (1 + gamma * m).
    """

    def __init__(
        self,
        d_model: int,
        num_patterns: int = 8,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_patterns = num_patterns
        self.d_context = max(16, d_model // 4)
        k1, k2 = 3, 7
        self.conv_k1 = nn.Conv1d(
            d_model, d_model, k1, padding="same", groups=d_model, bias=True
        )
        self.conv_k2 = nn.Conv1d(
            d_model, d_model, k2, padding="same", groups=d_model, bias=True
        )
        ctx_in = 2 * d_model

        self.context_mlp = nn.Sequential(
            nn.Linear(ctx_in, self.d_context),
            nn.GELU(),
        )
        self.route_proj = nn.Linear(self.d_context, num_patterns)
        self.mode_table = nn.Parameter(torch.empty(num_patterns, d_model))
        self.gamma = nn.Parameter(torch.zeros(1))

        nn.init.normal_(self.mode_table, std=0.02)
        for m in (self.conv_k1, self.conv_k2):
            nn.init.kaiming_normal_(m.weight, nonlinearity="linear")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        nn.init.normal_(self.route_proj.weight, std=0.02)
        nn.init.zeros_(self.route_proj.bias)
        nn.init.normal_(self.context_mlp[0].weight, std=0.02)
        nn.init.zeros_(self.context_mlp[0].bias)

    def forward(
        self,
        x: torch.Tensor,
        return_aux: bool = False,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]]:
        """
        Args:
            x: [B, L, d_model]
            return_aux: If True, also return dict with routing_probs and optional orth term.

        Returns:
            Modulated x of shape [B, L, d_model], optionally with aux dict.
        """
        b, l, d = x.shape
        if d != self.d_model:
            raise ValueError(f"Expected d_model={self.d_model}, got {d}")

        xt = x.transpose(1, 2)  # [B, d, L]
        y1 = self.conv_k1(xt)
        y2 = self.conv_k2(xt)
        z = torch.cat([y1, y2], dim=1)  # [B, 2d, L]
        z = z.transpose(1, 2)  # [B, L, 2d]
        c = self.context_mlp(z)  # [B, L, d_c]
        logits = self.route_proj(c)  # [B, L, P]
        p = F.softmax(logits, dim=-1)  # [B, L, P]
        m = torch.einsum("blk,kd->bld", p, self.mode_table)
        out = x * (1.0 + self.gamma * m)

        if not return_aux:
            return out

        aux: Dict[str, torch.Tensor] = {"routing_probs": p.detach()}
        return out, aux
