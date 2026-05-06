from typing import TYPE_CHECKING
import numpy as np
import torch

from .callback import BasicTSCallback

if TYPE_CHECKING:
    from basicts.runners.basicts_runner import BasicTSRunner


class LogDPRRouting(BasicTSCallback):
    """Log DPR routing probability statistics during training."""

    def __init__(self, log_interval: int = 1):
        super().__init__()
        self.log_interval = log_interval
        self._hooks = []
        self.routing_stats = []  # Per-batch routing tensors accumulated over the epoch

    def on_train_start(self, runner: "BasicTSRunner", *args, **kwargs) -> None:
        self._patch_dpr(runner.model)

    def _patch_dpr(self, model):
        from basicts.modules.dpr import TemporalContextualGating

        for name, module in model.named_modules():
            if isinstance(module, TemporalContextualGating):
                orig_forward = module.forward

                def make_patched_forward(orig):
                    def patched(x, return_aux=False):
                        result, aux = orig(x, return_aux=True)
                        if isinstance(aux, dict) and "routing_probs" in aux:
                            # Move each batch to CPU and append (avoid GPU memory growth)
                            self.routing_stats.append(aux["routing_probs"].detach().cpu())
                            if return_aux:
                                return result, aux
                        return result
                    return patched

                module.forward = make_patched_forward(orig_forward)
                self._hooks.append(module)

    def on_epoch_end(self, runner: "BasicTSRunner", epoch: int, *args, **kwargs) -> None:
        if epoch % self.log_interval != 0:
            return

        if not self.routing_stats:
            runner.logger.info(f"[Epoch {epoch}] DPR Routing: No routing captured yet")
            return

        # Concatenate all batches in the epoch; shape [Total_B, L, P]
        routing = torch.cat(self.routing_stats, dim=0).numpy()
        self.routing_stats.clear()  # Reset for the next epoch

        # --- 1. Temporal switching statistics (main metrics) ---

        # 1.a Hard switching rate
        # Top-1 pattern index per timestep: [Total_B, L]
        hard_choices = routing.argmax(axis=-1)
        # Whether pattern differs between adjacent timesteps: [Total_B, L-1]
        switches = (hard_choices[:, 1:] != hard_choices[:, :-1]).astype(float)
        switch_rate = switches.mean()  # ~0%: stable pattern; ~100%: frequent oscillation

        # 1.b Soft temporal change (L1 distance between consecutive routing vectors)
        # Sum absolute differences over P for each adjacent timestep pair
        soft_diff = np.abs(routing[:, 1:, :] - routing[:, :-1, :]).sum(axis=-1)
        mean_soft_diff = soft_diff.mean()  # Upper bound 2.0 for orthogonal one-hot switches

        # --- 2. Pattern usage balance and routing confidence (legacy metrics) ---

        # [Total_B, L, P] -> per-timestep entropy: [Total_B, L]
        per_timestep_entropy = -(routing * np.log(routing + 1e-8)).sum(-1)
        timestep_entropy_mean = per_timestep_entropy.mean()

        max_entropy = np.log(routing.shape[-1])
        per_sample_max_prob = routing.max(axis=-1).mean(axis=1)  # [Total_B]
        consistency = (per_sample_max_prob > 0.8).mean()

        usage_per_pattern = routing.mean(axis=(0, 1))
        usage_std = usage_per_pattern.std()

        runner.logger.info(f"[Epoch {epoch}] DPR Dynamic Switching Stats:")
        runner.logger.info(f"  --> Hard Switch Rate: {switch_rate*100:.2f}% of timesteps change active pattern.")
        runner.logger.info(f"  --> Soft Temporal Diff (L1): {mean_soft_diff:.4f} (Max 2.0)")
        runner.logger.info(f"[Epoch {epoch}] DPR Confidence & Balance Stats:")
        runner.logger.info(f"  Uniformity (entropy/max): {timestep_entropy_mean/max_entropy:.4f}")
        runner.logger.info(f"  Routing confidence (>0.8): {consistency*100:.2f}%")
        runner.logger.info(f"  Usage std across patterns: {usage_std:.4f}")
        runner.logger.info(f"  Per-pattern usage: {[round(x, 4) for x in usage_per_pattern.tolist()]}")

    def on_train_end(self, runner: "BasicTSRunner", *args, **kwargs) -> None:
        self._restore_dpr()

    def _restore_dpr(self):
        for module in self._hooks:
            if hasattr(module, '_orig_forward'):
                module.forward = module._orig_forward
        self._hooks.clear()