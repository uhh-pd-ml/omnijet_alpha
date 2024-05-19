# setup logging in notebook
import argparse
import logging
import sys
from pathlib import Path

import hydra
import lightning as L
import torch
from omegaconf import OmegaConf

import gabbro.plotting.utils as plot_utils
from gabbro.callbacks.classifier_callback import ClassifierEvaluationCallback

# from gabbro.models.classifiers import ClassifierPL
from gabbro.models.backbone import BackboneClassificationLightning
from gabbro.utils.pylogger import get_pylogger

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate tokenization ckpt")
    parser.add_argument(
        "--ckpt_path",
        type=Path,
        help="Path to the checkpoint to evaluate",
    )
    # parser.add_argument(
    #     "--n_eval",
    #     type=int,
    #     default=100_000,
    #     help="Number of jets to evaluate per jet type",
    # )
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Run in development mode (i.e. with less data)",
        default=False,
    )
    return parser.parse_args()


def main(ckpt_path, dev):
    plot_utils.set_mpl_style()
    logger = get_pylogger(__name__)
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    # complete the path
    ckpt_path = list(ckpt_path.parent.glob(ckpt_path.name))[0]
    config_path = ckpt_path.parent.parent / "config.yaml"
    cfg = OmegaConf.load(config_path)
    # cfg.data.dataset_kwargs_common["n_jets_per_file"] = n_eval
    logger.info(f"Loading config from: {config_path}")

    if dev:
        tbqq_cfg = cfg.data.dataset_kwargs_test.files_dict["Tbqq"]
        cfg.data.dataset_kwargs_test.files_dict = {"Tbqq": tbqq_cfg}

    # set number of files per jet type to test with
    cfg.data.dataset_kwargs_test.max_n_files_per_type = 10

    out_dir = ckpt_path.parent.parent / "evaluated_ckpts" / (ckpt_path.name).split("_loss_")[0]
    out_dir.mkdir(parents=True, exist_ok=True)
    # out_plot_dir = out_dir / "plots"
    # out_plot_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {out_dir}")

    # initialize the datamodule and and load the model from checkpoint
    datamodule = hydra.utils.instantiate(cfg.data)
    # model = ClassifierPL.load_from_checkpoint(ckpt_path)
    model = BackboneClassificationLightning.load_from_checkpoint(ckpt_path)
    model.eval()
    print(model)

    # initialize the trainer to run the test loop
    trainer = L.Trainer(enable_progress_bar=True)
    torch.set_float32_matmul_precision

    # ----------------------------------------------------------
    # --- TEST LOOP ---
    logger.info("----------- Starting test loop -----------")
    trainer.test(model, datamodule=datamodule)
    trainer.datamodule = datamodule

    callback = ClassifierEvaluationCallback(
        every_n_epochs=1,
        image_path=str(out_dir),
    )
    callback.on_test_end(trainer, model)


if __name__ == "__main__":
    args = parse_args()
    main(
        ckpt_path=args.ckpt_path,
        # n_eval=args.n_eval,
        dev=args.dev,
    )
