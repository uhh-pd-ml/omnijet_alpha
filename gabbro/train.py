import hashlib
import os
import time
from pathlib import Path
from typing import List, Optional, Tuple

import hydra
import lightning as L
import pyrootutils
import torch
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig, OmegaConf

import gabbro.utils.git_utils as git_utils
from gabbro.models.backbone import (
    BackboneClassificationLightning,
    BackboneNextTokenPredictionLightning,
)
from gabbro.models.classifiers import ClassifierPL
from gabbro.models.vqvae import VQVAELightning
from gabbro.utils.bigram import get_bigram
from gabbro.utils.pylogger import get_pylogger
from gabbro.utils.utils import (
    instantiate_callbacks,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
)

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/pyrootutils
# ------------------------------------------------------------------------------------ #


log = get_pylogger(__name__)


def get_nodename_bigram():
    """Generate a unique run identifier based on the nodename and a random bigram.
    Example: `max-wng029_QuickBear`

    The bigram is generated from the nodename and the current time, which means
    that two runs starting at the same time on different nodes will have different
    bigrams (if the nodename is not included, two runs starting at the same time
    will have the same bigram).

    Returns:
        str: Unique run identifier.
    """
    nodename = os.uname().nodename
    # cleanup
    nodename = nodename.split(".")[0]

    nodename_with_time = f"{nodename}_{int(time.time())}"

    # get a hash of the nodename
    hashed_nodename_with_time = hashlib.sha256(nodename_with_time.encode()).hexdigest()

    # bigram
    bigram = get_bigram(seed=int(hashed_nodename_with_time, 16))

    return "_".join([nodename, bigram])


# this is how we can include this resolver in the run directory (see configs/hydra/defaults.yaml)
OmegaConf.register_new_resolver("nodename_bigram", get_nodename_bigram, use_cache=True)


@task_wrapper
def train(cfg: DictConfig) -> Tuple[dict, dict]:
    """Trains the model. Can additionally evaluate on a testset, using best weights obtained during
    training.

    This method is wrapped in optional @task_wrapper decorator which applies extra utilities
    before and after the call.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.

    Returns:
        Tuple[dict, dict]: Dict with metrics and dict with all instantiated objects.
    """

    # check if cuda available
    if not torch.cuda.is_available():
        raise ValueError("CUDA is not available!")
    else:
        log.info("CUDA is available.")

    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    log.info(f"Git Status: {git_utils.get_git_status()}")
    log.info(f"Git Hash: {git_utils.get_git_hash()}")
    log.info(f"Last Commit Message: {git_utils.get_last_commit_message()}")

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: L.LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: L.LightningModule = hydra.utils.instantiate(cfg.model)

    # TODO: this is a bit of a hack, but it works for now (used to load the
    # VQVAE encoder for the classifier)

    model_class_name = cfg.model.get("model_class_name", None)

    if cfg.get("load_weights_from", False):
        log.info(f"Loading model weights from {cfg.load_weights_from}")

        load_cpt_path = Path(cfg.load_weights_from).parent.parent / "config.yaml"
        print("Model config before loading weights:")
        print(OmegaConf.to_yaml(cfg.model))
        cfg_ckpt = OmegaConf.load(load_cpt_path)
        cfg.model.model_kwargs_loaded = cfg_ckpt.model.model_kwargs

        if isinstance(model, VQVAELightning):
            model = VQVAELightning.load_from_checkpoint(
                cfg.load_weights_from,
                strict=cfg.get("load_weights_strict", False),
            )
        elif isinstance(model, ClassifierPL):
            model = ClassifierPL.load_from_checkpoint(
                cfg.load_weights_from,
                model_class_name=model_class_name,
                strict=cfg.get("load_weights_strict", False),
            )
        elif isinstance(model, BackboneNextTokenPredictionLightning):
            model = BackboneNextTokenPredictionLightning.load_from_checkpoint(
                cfg.load_weights_from,
                strict=cfg.get("load_weights_strict", False),
            )
        elif isinstance(model, BackboneClassificationLightning):
            model = BackboneClassificationLightning.load_from_checkpoint(
                cfg.load_weights_from,
                strict=cfg.get("load_weights_strict", False),
            )
        else:
            raise ValueError("Model not recognized!")
        print("Model config after loading weights:")
        print(OmegaConf.to_yaml(cfg.model))

    if cfg.model.model_kwargs.get("class_head_kwargs", False):
        model.model.class_head_kwargs = cfg.model.model_kwargs.class_head_kwargs
        model.model.initialize_classification_head()

    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info("Instantiating callbacks...")
    callbacks: List[L.Callback] = instantiate_callbacks(cfg.get("callbacks"))

    log.info(f"Model: \n{model}")

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: L.Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
        "git": {
            "git_hash": git_utils.get_git_hash(),
            "git_status": git_utils.get_git_status(),
            "git_last_commit_message": git_utils.get_last_commit_message(),
        },
        "slurm": {
            "job_id": os.environ.get("SLURM_JOB_ID", None),
            "log_file": os.environ.get("SLURM_LOGFILE", None),
        },
        "load_weights_from": cfg.get("load_weights_from", None),
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    # save config for reproducibility
    cfg_backup_file = f'{cfg.trainer.get("default_root_dir")}/config.yaml'
    with open(cfg_backup_file, "w") as f:
        log.info(f"Saving config to {cfg_backup_file}")
        OmegaConf.save(cfg, f)

    if cfg.get("train"):
        log.info("------------------")
        log.info("Starting training!")
        log.info("------------------")
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))

    if cfg.get("test"):
        log.info("-----------------")
        log.info("Starting testing!")
        log.info("-----------------")
        ckpt_path = trainer.checkpoint_callback.best_model_path
        if ckpt_path == "":
            log.warning("Best ckpt not found! Using current weights for testing...")
            ckpt_path = None
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        log.info(f"Best ckpt path: {ckpt_path}")

    return None


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    # set CUDA_LAUNCH_BLOCKING=1 to get more informative stack traces
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    torch.set_float32_matmul_precision("medium")

    experiment_name = Path(cfg.trainer.default_root_dir).name.split("_")[3]
    cfg.logger.comet.experiment_name = experiment_name
    cfg.logger.wandb.name = experiment_name

    # train the model
    train(cfg)

    return None


if __name__ == "__main__":
    main()
