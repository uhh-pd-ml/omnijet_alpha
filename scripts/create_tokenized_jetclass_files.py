import argparse
import glob
import logging
import os
from pathlib import Path

import dotenv

from gabbro.data.data_tokenization import tokenize_jetclass_file
from gabbro.utils.pylogger import get_pylogger

log = get_pylogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt_path", type=str, required=True)
parser.add_argument("--n_files_train", type=int, required=False, default=100)
parser.add_argument("--n_files_val", type=int, required=False, default=5)
parser.add_argument("--n_files_test", type=int, required=False, default=20)
parser.add_argument("--output_suffix", type=str, default="")
parser.add_argument("--dry_run", action="store_true")


def main(
    ckpt_path,
    jet_types,
    n_files_train,
    n_files_val,
    n_files_test,
    output_suffix="",
):
    """Tokenize the JetClass files and save them to the tokenized dir.

    Parameters
    ----------
    ckpt_path : str
        Path to the checkpoint to use for tokenization.
    jet_types : list of str
        List of jet types to use for tokenization.
    n_files_train : int
        Number of files to use for training.
    n_files_val : int
        Number of files to use for validation.
    n_files_test : int
        Number of files to use for testing.
    output_suffix : str
        Suffix to add to the output folder name.
    """

    log.info("Starting tokenization of JetClass files")
    log.info(f"Using checkpoint: {ckpt_path}")

    JETCLASS_DIR = dotenv.dotenv_values()["JETCLASS_DIR"]
    JETCLASS_DIR_TRAIN = Path(JETCLASS_DIR) / "train_100M"
    JETCLASS_DIR_VAL = Path(JETCLASS_DIR) / "val_5M"
    JETCLASS_DIR_TEST = Path(JETCLASS_DIR) / "test_20M"

    JETCLASS_DIR_TOKENIZED = Path(dotenv.dotenv_values()["JETCLASS_DIR_TOKENIZED"])
    run_id = Path(ckpt_path).parent.parent.name
    JETCLASS_DIR_TOKENIZED = JETCLASS_DIR_TOKENIZED / (str(run_id) + output_suffix)
    JETCLASS_DIR_TOKENIZED_TRAIN = JETCLASS_DIR_TOKENIZED / "train_100M"
    JETCLASS_DIR_TOKENIZED_VAL = JETCLASS_DIR_TOKENIZED / "val_5M"
    JETCLASS_DIR_TOKENIZED_TEST = JETCLASS_DIR_TOKENIZED / "test_20M"

    # raise error if tokenized dir already exists
    if JETCLASS_DIR_TOKENIZED.exists():
        raise FileExistsError(
            f"Output folder already exists"
            " - please delete it before running this script."
            f"\n\nrm -rf {JETCLASS_DIR_TOKENIZED}\n"
        )

    # create tokenized dirs
    JETCLASS_DIR_TOKENIZED_TRAIN.mkdir(parents=True, exist_ok=True)
    JETCLASS_DIR_TOKENIZED_VAL.mkdir(parents=True, exist_ok=True)
    JETCLASS_DIR_TOKENIZED_TEST.mkdir(parents=True, exist_ok=True)

    files_train = []
    files_val = []
    files_test = []

    for jt in jet_types:
        wildcard_train = f"{JETCLASS_DIR_TRAIN}/{jt}*.root"
        wildcard_val = f"{JETCLASS_DIR_VAL}/{jt}*.root"
        wildcard_test = f"{JETCLASS_DIR_TEST}/{jt}*.root"

        files_train.extend(sorted(list(glob.glob(wildcard_train)))[:n_files_train])
        files_val.extend(sorted(list(glob.glob(wildcard_val)))[:n_files_val])
        files_test.extend(sorted(list(glob.glob(wildcard_test)))[:n_files_test])

    log.info(f"Found {len(files_train)} train files:")
    for f in files_train:
        log.info(f)
    log.info(f"Found {len(files_val)} val files:")
    for f in files_val:
        log.info(f)
    log.info(f"Found {len(files_test)} test files:")
    for f in files_test:
        log.info(f)

    files_dict = {
        "train": files_train,
        "val": files_val,
        "test": files_test,
    }
    out_dirs = {
        "train": JETCLASS_DIR_TOKENIZED_TRAIN,
        "val": JETCLASS_DIR_TOKENIZED_VAL,
        "test": JETCLASS_DIR_TOKENIZED_TEST,
    }

    # copy the checkpoint to the tokenized dir
    os.system(f"cp {ckpt_path} {JETCLASS_DIR_TOKENIZED}")  # nosec
    os.system(f"cp {ckpt_path} {JETCLASS_DIR_TOKENIZED}/model_ckpt.ckpt")  # nosec
    cfg_path = Path(ckpt_path).parent.parent / "config.yaml"
    os.system(f"cp {cfg_path} {JETCLASS_DIR_TOKENIZED}")  # nosec

    for stage, files in files_dict.items():
        for i, filename_in in enumerate(files):
            log.info(f"{stage} file {i + 1}/{len(files)}")
            filename_out = Path(out_dirs[stage]) / Path(filename_in).name.replace(
                ".root", "_tokenized.parquet"
            )
            log.info("Input file: %s", filename_in)
            log.info("Output file: %s", filename_out)
            log.info("---")
            tokenize_jetclass_file(
                filename_in=filename_in,
                model_ckpt_path=ckpt_path,
                filename_out=filename_out,
                add_start_end_tokens=True,
            )


if __name__ == "__main__":
    args = parser.parse_args()
    ckpt_path = args.ckpt_path
    # jet_types = ["TTBar_", "ZJetsToNuNu_"]
    jet_types = [
        "ZJetsToNuNu_",
        "HToBB_",
        "HToCC_",
        "HToGG_",
        "HToWW4Q_",
        "HToWW2Q1L_",
        "ZToQQ_",
        "WToQQ_",
        "TTBar_",
        "TTBarLep_",
    ]
    n_files_train = args.n_files_train
    n_files_val = args.n_files_val
    n_files_test = args.n_files_test
    output_suffix = args.output_suffix
    if args.dry_run:
        log.info("Dry run - not actually running tokenization")
        log.info(f"Using checkpoint: {ckpt_path}")
        log.info(f"Jet types: {jet_types}")
        log.info(f"Output suffix: {output_suffix}")
        log.info(f"n_files_train: {n_files_train}")
        log.info(f"n_files_val: {n_files_val}")
        log.info(f"n_files_test: {n_files_test}")
        exit(0)
    main(
        ckpt_path,
        jet_types,
        n_files_train,
        n_files_val,
        n_files_test,
        output_suffix,
    )
