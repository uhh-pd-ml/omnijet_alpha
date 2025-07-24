import glob
import hashlib
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import comet_ml  # noqa: F401  # we need to import comet_ml before torch for correct logging
import hydra
import lightning as L
import pyrootutils
import torch
from hydra import compose
from hydra.core.config_store import ConfigStore
from hydra.core.hydra_config import HydraConfig
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig, OmegaConf
from torch.distributed import get_rank, get_world_size

import gabbro.models.lightning_models as gabbro_lightning_models
import gabbro.utils.git_utils as git_utils
from gabbro.utils.bigram import get_bigram
from gabbro.utils.pylogger import get_pylogger
from gabbro.utils.utils import (
    get_gpu_properties,
    instantiate_callbacks,
    instantiate_loggers,
    log_hyperparameters,
    remove_empty_hydra_run_dir,
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

    If a job ID exists (this needs to be passed to the container in the job
    submission script via eg. --env JOB_ID="$SLURM_JOB_ID"), the bigram will be
    seeded based on this ID. This means that a multi-node run will have the same
    bigram, which is useful when the processes on one node need to access files
    in the directory belonging to the main node containing rank 0.

    If no job ID exists, the bigram is generated from the nodename and the
    current time, which means that two runs starting at the same time on
    different nodes will have different bigrams (if the nodename is not included,
    two runs starting at the same time will have the same bigram).

    Returns:
        str: Unique run identifier.
    """
    nodename = os.uname().nodename
    job_id = os.environ.get("JOB_ID", None)

    # cleanup
    nodename = nodename.split(".")[0]

    nodename_with_time = f"{nodename}_{int(time.time())}"

    # get hashes
    if job_id is not None:
        log.info(f"Job ID {job_id} detected. Generating bigram based on this job ID.")
        hashed_name = hashlib.sha256(job_id.encode()).hexdigest()
    else:
        log.info(
            f"No job ID detected. Generating bigram based on node name and time stamp, {nodename_with_time}."
        )
        hashed_name = hashlib.sha256(nodename_with_time.encode()).hexdigest()

    # bigram
    bigram = get_bigram(seed=int(hashed_name, 16))

    if job_id is not None:
        return "_".join([nodename, bigram, job_id])
    else:
        return "_".join([nodename, bigram])


# this is how we can include this resolver in the run directory (see configs/hydra/defaults.yaml)
OmegaConf.register_new_resolver("nodename_bigram", get_nodename_bigram, use_cache=True)
# add eval resolver to evaluate expressions in the config
OmegaConf.register_new_resolver("eval", eval)


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

    if cfg.get("continue_from_checkpoint", False):
        ckpt_path = Path(cfg.continue_from_checkpoint)
        log.info(f"Loading model from lightning checkpoint {ckpt_path}")
        model_class_name = cfg.model._target_.split(".")[-1]
        try:
            lightning_module_class = getattr(gabbro_lightning_models, model_class_name)
        except AttributeError:
            raise AttributeError(
                f"Model class {model_class_name} not found in gabbro.lightning_models. "
                f"To be able to load a model from a checkpoint, the model class must be "
                f"imported in gabbro.lightning_models. "
                f"Available models are: {dir(gabbro_lightning_models)}"
            )
        model = lightning_module_class.load_from_checkpoint(ckpt_path)
    else:
        log.info(f"Instantiating model <{cfg.model._target_}>")
        model: L.LightningModule = hydra.utils.instantiate(cfg.model)

        if cfg.get("load_weights_from", False):
            log.info(f"Loading model weights from {cfg.load_weights_from}")

            load_cpt_path = Path(cfg.load_weights_from).parent.parent / "config.yaml"
            print("Model config before loading weights:")
            print(OmegaConf.to_yaml(cfg.model))
            cfg_ckpt = OmegaConf.load(load_cpt_path)
            cfg.model.model_kwargs_loaded = cfg_ckpt.model.model_kwargs

            # we want to only load the weights, not the optimizer state etc. as would
            # be done with LightningModule.load_from_checkpoint()
            state_dict = torch.load(cfg.load_weights_from, map_location="cpu")["state_dict"]  # nosec
            model.load_state_dict(state_dict, strict=cfg.get("load_weights_strict", True))

            log.info("Model config after loading weights:")
            log.info(OmegaConf.to_yaml(cfg.model))

    if cfg.model.model_kwargs.get("class_head_kwargs", False):
        model.model.class_head_kwargs = cfg.model.model_kwargs.class_head_kwargs
        model.model.initialize_classification_head()

    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info("Instantiating callbacks...")
    callbacks: Dict[str, L.Callback] = instantiate_callbacks(cfg.get("callbacks"))

    log.info("Done instantiating callbacks.")
    log.info("Callbacks:")
    for cb_name, cb in callbacks.items():
        log.info(f"- {cb_name}: {cb}")

    log.info(f"Model: \n{model}")

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: L.Trainer = hydra.utils.instantiate(
        cfg.trainer,
        callbacks=list(callbacks.values()),
        logger=logger,
    )

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
        "gpu_properties": get_gpu_properties(),
    }

    if logger and cfg.get("ckpt_path_for_evaluation") is None:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    if cfg.get("train"):
        # --- Save config for reproducibility --- #
        # save config
        cfg_backup_file = f'{cfg.trainer.get("default_root_dir")}/config.yaml'
        with open(cfg_backup_file, "w") as f:
            log.info(f"Saving config to {cfg_backup_file}")
            OmegaConf.save(cfg, f)
        # save resolved config
        cfg_resolved_file = f'{cfg.trainer.get("default_root_dir")}/config_resolved.yaml'
        with open(cfg_resolved_file, "w") as f:
            log.info(f"Saving resolved config to {cfg_resolved_file}")
            OmegaConf.save(cfg, f, resolve=True)
        # ---

        log.info("------------------")
        log.info("Starting training!")
        log.info("------------------")
        log.info(f"Global rank: {trainer.global_rank}")
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))

    if cfg.get("test"):
        log.info("-----------------")
        log.info("Starting testing!")
        log.info("-----------------")

        if cfg.get("ckpt_path_for_evaluation") is not None:
            # evaluate a specific checkpoint
            ckpt_path = cfg.get("ckpt_path_for_evaluation")
        else:
            # No specific checkpoint for evaluation was provided --> this is what happens
            # at the end of training when the model is tested on the best weights
            # --> we need to find the best ckpt in the callbacks. This may be in the
            # directory of another node (it only saves checkpoints on rank 0).

            # Keep track of which process gets which checkpoint
            process_rank = get_rank() if torch.distributed.is_initialized() else 0
            world_size = get_world_size() if torch.distributed.is_initialized() else 1

            # Check if the node's root directory contains checkpoints at all
            if os.path.isdir(f"{cfg.trainer.default_root_dir}/checkpoints"):
                name_best_ckpt = "model_checkpoint_best"
                name_ckpt = "model_checkpoint"
                if name_best_ckpt in callbacks:
                    ckpt_path = callbacks[name_best_ckpt].best_model_path
                    log.info(
                        f"Rank {process_rank}/[0-{world_size-1}]: Using best model path from callback {name_best_ckpt}: {ckpt_path}"
                    )
                # if best ckpt not found in that callback, try with other name
                elif name_ckpt in callbacks:
                    ckpt_path = callbacks[name_ckpt].best_model_path
                    log.info(
                        f"Rank {process_rank}/[0-{world_size-1}]: Using best model path from callback {name_ckpt}: {ckpt_path}"
                    )
                # if best ckpt not found there either, just use the last ckpt
                # which is stored separately as last.ckpt
                else:
                    log.warning(
                        f"Neither '{name_best_ckpt}' nor '{name_ckpt}' found in callbacks!"
                    )
                    log.warning(
                        f"Rank {process_rank}/[0-{world_size-1}]: Best ckpt not found! Using last.ckpt for testing..."
                    )
                    ckpt_path = f"{cfg.trainer.default_root_dir}/checkpoints/last.ckpt"
            else:
                # If the root directory does not have checkpoints, look for one that does
                root_dir = cfg.trainer.default_root_dir
                log.info(
                    f"Rank {process_rank}/[0-{world_size-1}]: The root directory {root_dir} does not contain checkpoints. Searching other directories..."
                )
                bigram = (root_dir.split("/")[-1]).split("_")[3]
                log.info(f"Extracted bigram {bigram}")
                # Get a list of all paths with this bigram in the parent directory of current run directory
                p = Path(root_dir)
                p = p.resolve()
                directory_list = glob.glob(f"{os.path.join(p.parent,'*'+str(bigram)+'*')}")
                directory_list.sort()
                log.info(f"List of directories containing the bigram {bigram}: {directory_list}")
                found_checkpoints = False
                for directory in directory_list:
                    # Check if it contains a checkpoint directory
                    if os.path.isdir(f"{directory}/checkpoints"):
                        checkpoint_directory = f"{directory}/checkpoints"
                        log.info(f"Checkpoint directory detected: {checkpoint_directory}")
                        # Now try to find the checkpoint
                        # -- best.ckpt
                        if os.path.isfile(f"{checkpoint_directory}/best.ckpt"):
                            ckpt_path = f"{checkpoint_directory}/best.ckpt"
                            log.info(
                                f"Rank {process_rank}/[0-{world_size-1}]: Using best checkpoint from {ckpt_path}"
                            )
                        # -- last.ckpt
                        elif os.path.isfile(f"{checkpoint_directory}/last.ckpt"):
                            ckpt_path = f"{checkpoint_directory}/last.ckpt"
                            log.info(
                                f"Rank {process_rank}/[0-{world_size-1}]: Using last checkpoint from {ckpt_path}"
                            )
                        # -- the very last .ckpt file in the list of all checkpoint files
                        else:
                            all_checkpoints = glob.glob(
                                f"{os.path.join(checkpoint_directory,'*.ckpt')}"
                            )
                            # If the directory does not have any .ckpt files
                            if len(all_checkpoints) == 0:
                                log.info(f"Can not find any checkpoints in {checkpoint_directory}")
                                continue
                            all_checkpoints.sort()
                            ckpt_path = all_checkpoints[-1]  # Take the last one
                            log.info(
                                f"Rank {process_rank}/[0-{world_size-1}]: Using last checkpoint from {ckpt_path}"
                            )
                        found_checkpoints = True
                # if we still couldn't find the checkpoint
                assert found_checkpoints, f"Rank {process_rank}/[0-{world_size-1}]: No checkpoints could be found, exiting."
        # ------------------------------------------------

        # update the default root dir for testing
        ckpt_filename = Path(ckpt_path).name
        cfg.trainer.default_root_dir = (
            Path(cfg.trainer.default_root_dir) / "evaluation" / ckpt_filename
        )
        cfg.trainer.default_root_dir.mkdir(parents=True, exist_ok=True)

        log.info(f"Set default_root_dir to {trainer.default_root_dir}")

        log.info(f"Instantiating trainer for testing <{cfg.trainer._target_}>")
        trainer = hydra.utils.instantiate(
            cfg.trainer,
            logger=logger if cfg.get("ckpt_path_for_evaluation") is None else None,
            callbacks=list(callbacks.values()),
        )

        if ckpt_path == "":
            log.warning(
                "Best ckpt either not found or not accessible in the callbacks! "
                "Using current weights for testing..."
            )
        log.info(f"Best ckpt path: {ckpt_path}")

        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        log.info(f"Testing was done with ckpt path: {ckpt_path}")

    return None


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    # set CUDA_LAUNCH_BLOCKING=1 to get more informative stack traces
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    torch.set_float32_matmul_precision("medium")

    experiment_name = Path(cfg.trainer.default_root_dir).name.split("_")[3]
    cfg.logger.comet.experiment_name = experiment_name
    cfg.logger.wandb.name = experiment_name

    # load full config from file if specified
    if cfg.get("ckpt_path_for_evaluation") is not None:
        ckpt_path = Path(cfg.ckpt_path_for_evaluation)
        log.info(f"Will evaluate the model checkpoint {ckpt_path}")

        cfg_ckpt_path = ckpt_path.parent.parent / "config.yaml"
        log.info(f"Loading config from {cfg_ckpt_path} for evaluation")
        cfg_ckpt = OmegaConf.load(cfg_ckpt_path)

        # get the redundant output dir created by hydra when this evaluation
        # was started (hydra always creates a new output dir for each execution
        # but we want to use the same output dir as the training run)
        redundant_output_dir = Path(cfg.paths.output_dir)
        # compose cfg from the new cfg_path + the overrides in the
        # redundant output dir / ".hydra" / "overrides.yaml" (cause those are the
        # ones passed in the command line)
        ConfigStore.instance().store("cfg_ckpt", node=cfg_ckpt)
        cfg = compose(config_name="cfg_ckpt", overrides=HydraConfig.get().overrides.task)

        # set the output dir to the parent of the ckpt config path
        log.info(f"Setting output dir to {cfg_ckpt_path.parent}")
        cfg.paths.output_dir = cfg_ckpt_path.parent

        remove_empty_hydra_run_dir(redundant_output_dir)

        log.info(f"paths.output_dir={cfg.paths.output_dir}")
        log.info(f"logger.comet.experiment_name={cfg.logger.comet.experiment_name}")

        # set the evaluation flag to True and the training flag to False
        log.info("Setting evaluation flag to True and training flag to False")
        cfg.train = False
        cfg.test = True
        # set to single-node, single-gpu strategy for evaluation (because this will
        # crash otherwise if the model is evaluated on a single GPU and was trained
        # on multiple GPUs)
        # log.info("Setting trainer to single-node, single-gpu strategy for evaluation")
        # cfg.trainer.num_nodes = 1
        # cfg.trainer.devices = 1
        # cfg.trainer.strategy = "auto"

    # train the model
    train(cfg)

    return None


if __name__ == "__main__":
    main()
