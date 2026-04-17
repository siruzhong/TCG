from typing import TYPE_CHECKING
import numpy as np
import torch

from .callback import BasicTSCallback

if TYPE_CHECKING:
    from basicts.runners.basicts_runner import BasicTSRunner


class LogTCGRouting(BasicTSCallback):
    """Log TCG routing probability statistics during training."""

    def __init__(self, log_interval: int = 1):
        super().__init__()
        self.log_interval = log_interval
        self._hooks = []
        self.routing_stats = []  # 用列表收集整个 Epoch 的数据

    def on_train_start(self, runner: "BasicTSRunner", *args, **kwargs) -> None:
        self._patch_tcg(runner.model)

    def _patch_tcg(self, model):
        from basicts.modules.tcg import TemporalContextualGating

        for name, module in model.named_modules():
            if isinstance(module, TemporalContextualGating):
                orig_forward = module.forward

                def make_patched_forward(orig):
                    def patched(x, return_aux=False):
                        result, aux = orig(x, return_aux=True)
                        if isinstance(aux, dict) and "routing_probs" in aux:
                            # 修复：将每个 batch 的数据转移到 CPU 并追加到列表中
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
            runner.logger.info(f"[Epoch {epoch}] TCG Routing: No routing captured yet")
            return

        # 修复：拼接整个 Epoch 的所有 Batch，形状为 [Total_B, L, P]
        routing = torch.cat(self.routing_stats, dim=0).numpy()
        self.routing_stats.clear()  # 清空列表，为下一个 Epoch 做准备

        # --- 1. 时序维度的切换统计 (核心指标) ---

        # 1.a 硬切换率 (Hard Switching Rate)
        # 获取每个时刻选中的 top-1 pattern: [Total_B, L]
        hard_choices = routing.argmax(axis=-1)
        # 比较相邻时刻 pattern 是否变化: [Total_B, L-1]
        switches = (hard_choices[:, 1:] != hard_choices[:, :-1]).astype(float)
        switch_rate = switches.mean()  # 0% 表示从不切换，接近 100% 表示频繁震荡

        # 1.b 软变化幅度 (Soft Temporal Difference - L1 Distance)
        # [Total_B, L-1, P] 算绝对差值的和
        soft_diff = np.abs(routing[:, 1:, :] - routing[:, :-1, :]).sum(axis=-1)
        mean_soft_diff = soft_diff.mean()  # 最大值为 2.0 (完全正交的切换)

        # --- 2. 模式的使用平衡度与置信度 (原始指标保留) ---

        # [Total_B, L, P] -> per-timestep entropy: [Total_B, L]
        per_timestep_entropy = -(routing * np.log(routing + 1e-8)).sum(-1)
        timestep_entropy_mean = per_timestep_entropy.mean()

        max_entropy = np.log(routing.shape[-1])
        per_sample_max_prob = routing.max(axis=-1).mean(axis=1)  # [Total_B]
        consistency = (per_sample_max_prob > 0.8).mean()

        usage_per_pattern = routing.mean(axis=(0, 1))
        usage_std = usage_per_pattern.std()

        runner.logger.info(f"[Epoch {epoch}] TCG Dynamic Switching Stats:")
        runner.logger.info(f"  --> Hard Switch Rate: {switch_rate*100:.2f}% of timesteps change active pattern.")
        runner.logger.info(f"  --> Soft Temporal Diff (L1): {mean_soft_diff:.4f} (Max 2.0)")
        runner.logger.info(f"[Epoch {epoch}] TCG Confidence & Balance Stats:")
        runner.logger.info(f"  Uniformity (entropy/max): {timestep_entropy_mean/max_entropy:.4f}")
        runner.logger.info(f"  Routing confidence (>0.8): {consistency*100:.2f}%")
        runner.logger.info(f"  Usage std across patterns: {usage_std:.4f}")
        runner.logger.info(f"  Per-pattern usage: {[round(x, 4) for x in usage_per_pattern.tolist()]}")

    def on_train_end(self, runner: "BasicTSRunner", *args, **kwargs) -> None:
        self._restore_tcg()

    def _restore_tcg(self):
        for module in self._hooks:
            if hasattr(module, '_orig_forward'):
                module.forward = module._orig_forward
        self._hooks.clear()