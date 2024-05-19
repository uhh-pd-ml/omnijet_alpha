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
        self.logger = get_pylogger(state_key)

    @property
    def state_key(self) -> str:
        return self._state_key

    def _save_checkpoint(self, trainer, filepath):
        self.logger.info(f"Saving checkpoint to {filepath}")
        super()._save_checkpoint(trainer, filepath)
