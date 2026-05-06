"""Dynamic Pattern Routing (DPR): adaptive per-position modulation via local dynamics."""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Union

import torch
import torch.nn.functional as F
from torch import nn


def dpr_orthogonal_loss(mode_table: torch.Tensor) -> torch.Tensor:
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
        conv_kernels: Optional[Sequence[int]] = None,
        identity_init: bool = True,
        discrete_topk: int = 1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_patterns = num_patterns
        self.use_multiscale = use_multiscale
        self.discrete_topk = discrete_topk
        self.d_context = max(16, d_model // 4)

        if conv_kernels is None:
            kernels = (3, 7) if use_multiscale else (1,)
        else:
            kernels = tuple(int(k) for k in conv_kernels)
            if len(kernels) == 0:
                raise ValueError("conv_kernels must not be empty")
            if any(k <= 0 for k in kernels):
                raise ValueError(f"conv_kernels must be positive, got {kernels}")
        self.conv_kernels = kernels

        self.conv_layers = nn.ModuleList(
            [
                nn.Conv1d(
                    d_model, d_model, k, padding="same", groups=d_model, bias=True
                )
                for k in self.conv_kernels
            ]
        )
        ctx_in = len(self.conv_layers) * d_model

        self.context_mlp = nn.Sequential(
            nn.Linear(ctx_in, self.d_context),
            nn.GELU(),
        )
        self.route_centroids = nn.Parameter(torch.randn(num_patterns, self.d_context))
        nn.init.normal_(self.route_centroids, std=0.02)
        self.routing_scale = nn.Parameter(torch.ones(1) * 2.0)
        self.mode_table = nn.Parameter(torch.empty(num_patterns, d_model))
        if identity_init:
            self.gamma = nn.Parameter(torch.ones(1) * 0.1)
        else:
            self.gamma = nn.Parameter(torch.randn(1) * 0.01)

        nn.init.normal_(self.mode_table, std=0.02)
        for conv in self.conv_layers:
            if conv is not None:
                nn.init.kaiming_normal_(conv.weight, nonlinearity="linear")
                if conv.bias is not None:
                    nn.init.zeros_(conv.bias)
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
        conv_outs = [conv(xt) for conv in self.conv_layers]
        if len(conv_outs) == 1:
            z = conv_outs[0]
        else:
            z = torch.cat(conv_outs, dim=1)
        z = z.transpose(1, 2)  # [B, L, 2d] or [B, L, d]
        c = self.context_mlp(z)  # [B, L, d_c]
        c_norm = F.normalize(c, dim=-1) # [B, L, d_c]
        cent_norm = F.normalize(self.route_centroids, dim=-1) # [K, d_c]
        logits = torch.einsum("bld,kd->blk", c_norm, cent_norm) * self.routing_scale
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
