from typing import TYPE_CHECKING, Iterable

from .callback import BasicTSCallback

if TYPE_CHECKING:
    from basicts.runners.basicts_runner import BasicTSRunner


class AddAuxiliaryLoss(BasicTSCallback):
    """
    Adding auxiliary loss callback.
    """

    def __init__(self, losses: Iterable[str] = None):
        """
        Args:
            losses: Iterable[str], keys of losses in `forward_return` that will be added. Default is ["aux_loss"].
        """
        super().__init__()
        self.losses = tuple(losses or ["aux_loss"])

    def on_train_start(self, runner: "BasicTSRunner", *args, **kwargs) -> None:
        merged = list(getattr(runner, "_auxiliary_loss_keys", []) or [])
        for name in self.losses:
            if name not in merged:
                merged.append(name)
        runner._auxiliary_loss_keys = merged

    def on_compute_loss(self, runner: "BasicTSRunner", **kwargs):
        """Reserved; auxiliary terms are summed in ``BasicTSRunner`` after the base metric."""
        pass
