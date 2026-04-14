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
        use_multiscale: bool = True,
        identity_init: bool = True,
        discrete_topk: int = 1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_patterns = num_patterns
        self.use_multiscale = use_multiscale
        self.discrete_topk = discrete_topk
        self.d_context = max(16, d_model // 4)

        if use_multiscale:
            k1, k2 = 3, 7
            self.conv_k1 = nn.Conv1d(
                d_model, d_model, k1, padding="same", groups=d_model, bias=True
            )
            self.conv_k2 = nn.Conv1d(
                d_model, d_model, k2, padding="same", groups=d_model, bias=True
            )
            ctx_in = 2 * d_model
        else:
            self.conv_k1 = nn.Conv1d(
                d_model, d_model, 1, padding="same", groups=d_model, bias=True
            )
            self.conv_k2 = None
            ctx_in = d_model

        self.context_mlp = nn.Sequential(
            nn.Linear(ctx_in, self.d_context),
            nn.GELU(),
        )
        self.route_proj = nn.Linear(self.d_context, num_patterns)
        self.mode_table = nn.Parameter(torch.empty(num_patterns, d_model))
        if identity_init:
            self.gamma = nn.Parameter(torch.zeros(1))
        else:
            self.gamma = nn.Parameter(torch.randn(1) * 0.01)

        nn.init.normal_(self.mode_table, std=0.02)
        for conv in (self.conv_k1, self.conv_k2):
            if conv is not None:
                nn.init.kaiming_normal_(conv.weight, nonlinearity="linear")
                if conv.bias is not None:
                    nn.init.zeros_(conv.bias)
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
        if self.use_multiscale:
            assert self.conv_k2 is not None
            y1 = self.conv_k1(xt)
            y2 = self.conv_k2(xt)
            z = torch.cat([y1, y2], dim=1)  # [B, 2d, L]
        else:
            z = self.conv_k1(xt)
        z = z.transpose(1, 2)  # [B, L, 2d] or [B, L, d]
        c = self.context_mlp(z)  # [B, L, d_c]
        logits = self.route_proj(c)  # [B, L, P]
        if self.discrete_topk > 1:
            k = self.discrete_topk
            topk_vals, topk_idx = torch.topk(logits, k, dim=-1)
            p = torch.zeros_like(logits).scatter_(-1, topk_idx, F.softmax(topk_vals, dim=-1))
        else:
            p = F.softmax(logits, dim=-1)  # [B, L, P]
        m = torch.einsum("blk,kd->bld", p, self.mode_table)
        out = x * (1.0 + self.gamma * m)

        if not return_aux:
            return out

        aux: Dict[str, torch.Tensor] = {"routing_probs": p.detach()}
        return out, aux
