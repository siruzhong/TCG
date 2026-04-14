"""Hyperparameters for Temporal-Contextual Gating (TCG)."""

from dataclasses import dataclass
from typing import Optional

from basicts.modules.tcg import TemporalContextualGating


@dataclass
class TCGConfig:
    """When ``enabled`` is False, models skip building TCG and related losses.

    Ablation flags (default True/1 preserves original behavior):
        use_multiscale: Use k1=3, k2=7 depthwise convs (False -> k=1 point-wise only)
        identity_init: Initialize gamma=0 (False -> gamma ~ N(0, 0.01))
        discrete_topk: Soft routing (1) vs discrete Top-K routing (>1, e.g. 2 for Top-2)
    """

    enabled: bool = False
    num_patterns: int = 8
    orth_lambda: float = 0.01
    use_multiscale: bool = True
    identity_init: bool = True
    discrete_topk: int = 1

    def build_module(self, d_model: int) -> Optional[TemporalContextualGating]:
        if not self.enabled:
            return None
        return TemporalContextualGating(
            d_model=d_model,
            num_patterns=self.num_patterns,
            use_multiscale=self.use_multiscale,
            identity_init=self.identity_init,
            discrete_topk=self.discrete_topk,
        )
