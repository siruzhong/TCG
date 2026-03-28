"""Hyperparameters for Temporal-Contextual Gating (TCG)."""

from dataclasses import dataclass
from typing import Optional

from basicts.modules.tcg import TemporalContextualGating


@dataclass
class TCGConfig:
    """When ``enabled`` is False, models skip building TCG and related losses."""

    enabled: bool = False
    num_patterns: int = 8
    orth_lambda: float = 0.01

    def build_module(self, d_model: int) -> Optional[TemporalContextualGating]:
        if not self.enabled:
            return None
        return TemporalContextualGating(
            d_model=d_model,
            num_patterns=self.num_patterns,
        )
