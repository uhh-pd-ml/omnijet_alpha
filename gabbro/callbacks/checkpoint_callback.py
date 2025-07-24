from lightning.pytorch.callbacks import ModelCheckpoint

from gabbro.utils.pylogger import get_pylogger


class CustomModelCheckpoint(ModelCheckpoint):
    """Custom ModelCheckpoint callback that allows to specify the state_key to be used for the best
    checkpoint.

    This workaround is needed because it's not allowed to have two ModelCheckpoint callbacks with
    the same state_key in the same Trainer.
    """

    def __init__(self, state_key="best_checkpoint", **kwargs):
        super().__init__(**kwargs)
        self._state_key = state_key
        self.pylogger = get_pylogger(self._state_key)

    @property
    def state_key(self) -> str:
        return self._state_key

    def on_train_start(self, trainer, pl_module):
        super().on_train_start(trainer, pl_module)
        self.pylogger.info("`on_train_start` called.")
        self.pylogger.info("Setting up the logger with the correct rank.")
        self.pylogger = get_pylogger(self._state_key, rank=trainer.global_rank)
        self.pylogger.info("Logger set up.")

    def _save_checkpoint(self, trainer, filepath):
        self.pylogger.info(f"Saving checkpoint to {filepath} at rank {trainer.global_rank}")
        if trainer.global_rank != 0:
            self.pylogger.info(
                "save_checkpoint() is called at rank != 0, which will be skipped under "
                "the hood, so you won't find the file in the directory."
            )
        super()._save_checkpoint(trainer, filepath)
