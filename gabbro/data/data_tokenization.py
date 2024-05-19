import logging
from pathlib import Path

import awkward as ak
import numpy as np
import vector
from omegaconf import OmegaConf

from gabbro.data.loading import read_jetclass_file
from gabbro.models.vqvae import VQVAELightning
from gabbro.utils.jet_types import get_jet_type_from_file_prefix, jet_types_dict

vector.register_awkward()

logger = logging.getLogger(__name__)


def tokenize_jetclass_file(
    filename_in: str,
    model_ckpt_path: str,
    filename_out: str = None,
    add_start_end_tokens: bool = False,
    print_model: bool = False,
):
    """Tokenize a single file using a trained model.

    Parameters
    ----------
    filename : str
        Path to the file to be tokenized.
    model_ckpt_path : str
        Path to the model checkpoint.
    filename_out : str, optional
        Path to the output file.
    add_start_end_tokens : bool, optional
        Whether to add start and end tokens to the tokenized sequence.
    print_model : bool, optional
        Whether to print the model architecture.

    Returns
    -------
    tokens_int : ak.Array
        Array of tokens.
    p4s_original : ak.Array
        Momentum4D array of the original particles.
    x_ak_original : ak.Array
        Array of the original particles.
    """
    x_ak, _, _ = read_jetclass_file(
        filename_in,
        particle_features=["part_pt", "part_etarel", "part_phirel"],
        jet_features=None,
        return_p4=False,
    )

    # --- Model and config loading ---
    ckpt_path = Path(model_ckpt_path)
    config_path = ckpt_path.parent.parent / "config.yaml"
    cfg = OmegaConf.load(config_path)
    logger.info(f"Loaded config from {config_path}")
    model = VQVAELightning.load_from_checkpoint(ckpt_path)
    if print_model:
        print(model)
    pp_dict = cfg.data.dataset_kwargs_common["feature_dict"]
    logger.info("Preprocessing dictionary:")
    for key, value in pp_dict.items():
        logger.info(f" | {key}: {value}")

    model = model.to("cuda")
    model.eval()
    # --------------------------------

    p4s_original = ak.zip(
        {
            "pt": x_ak.part_pt,
            "eta": x_ak.part_etarel,
            "phi": x_ak.part_phirel,
            "mass": ak.zeros_like(x_ak.part_pt),
        },
        with_name="Momentum4D",
    )
    tokens = model.tokenize_ak_array(x_ak, pp_dict)

    if add_start_end_tokens:
        n_tokens = model.model.vqlayer.num_codes
        tokens = ak.concatenate(
            [
                ak.zeros_like(tokens[:, :1]),  # start token is 0
                tokens + 1,
                ak.ones_like(tokens[:, :1]) + n_tokens,  # end token is n_tokens + 1
            ],
            axis=1,
        )

    tokens_int = ak.values_astype(tokens, int)

    if filename_out is not None:
        logger.info(f"Saving tokenized file to {filename_out}")
        ak.to_parquet(tokens_int, filename_out)

    return tokens_int, p4s_original, x_ak


def reconstruct_jetclass_file(
    filename_in: str,
    model_ckpt_path: str,
    config_path: str,
    filename_out: str = None,
    start_token_included: bool = False,
    end_token_included: bool = False,
    shift_tokens_by_minus_one: bool = False,
    print_model: bool = False,
    device: str = "cuda",
    return_labels: bool = False,
):
    """Reconstruct a single file using a trained model and the tokenized file.

    Parameters
    ----------
    filename_in : str
        Path to the file to be tokenized.
    model_ckpt_path : str
        Path to the model checkpoint.
    config_path : str
        Path to the config file.
    filename_out : str, optional
        Path to the output file.
    start_token_included : bool, optional
        Whether the start token is included in the tokenized sequence.
    end_token_included : bool, optional
        Whether the end token is included in the tokenized sequence.
    shift_tokens_by_minus_one : bool, optional
        Whether to shift the tokens by -1.
    print_model : bool, optional
        Whether to print the model architecture.
    device : str, optional
        Device to use for the model.
    return_labels : bool, optional
        Whether to return the labels of the jet type. By default, the labels are not returned.

    Returns
    -------
    p4s_reco : ak.Array
        Momentum4D array of the reconstructed particles.
    x_reco_ak : ak.Array
        Array of the reconstructed particles.
    labels_onehot : np.ndarray
        One-hot encoded labels of the jet type. Only returned if return_labels is True.
    """

    # --- Model and config loading ---
    ckpt_path = Path(model_ckpt_path)
    cfg = OmegaConf.load(config_path)
    logger.info(f"Loaded config from {config_path}")
    model = VQVAELightning.load_from_checkpoint(ckpt_path)
    if print_model:
        print(model)
    pp_dict = cfg.data.dataset_kwargs_common["feature_dict"]
    # logger.info("Preprocessing dictionary:")
    # for key, value in pp_dict.items():
    #     logger.info(f" | {key}: {value}")

    model = model.to(device)
    model.eval()
    # --------------------------------

    tokens = ak.from_parquet(filename_in)
    if end_token_included:
        logger.info("Removing end token")
        tokens = tokens[:, :-1]
    if start_token_included:
        logger.info("Removing start token and shifting tokens by -1")
        tokens = tokens[:, 1:]
    if shift_tokens_by_minus_one:
        logger.info("Shifting tokens by -1")
        tokens = tokens - 1

    logger.info(f"Smallest token in file: {ak.min(tokens)}")
    logger.info(f"Largest token in file:  {ak.max(tokens)}")

    x_reco_ak = model.reconstruct_ak_tokens(tokens, pp_dict, hide_pbar=True)

    p4s_reco = ak.zip(
        {
            "pt": x_reco_ak.pt if "pt" in x_reco_ak.fields else x_reco_ak.part_pt,
            "eta": x_reco_ak.etarel if "etarel" in x_reco_ak.fields else x_reco_ak.part_etarel,
            "phi": x_reco_ak.phirel if "phirel" in x_reco_ak.fields else x_reco_ak.part_phirel,
            "mass": ak.zeros_like(x_reco_ak.pt if "pt" in x_reco_ak.fields else x_reco_ak.part_pt),
        },
        with_name="Momentum4D",
    )
    if return_labels:
        # extract jet type from filename and create the corresponding labels
        jet_type_prefix = filename_in.split("/")[-1].split("_")[0] + "_"
        jet_type_name = get_jet_type_from_file_prefix(jet_type_prefix)

        # one-hot encode the jet type
        labels_onehot = ak.Array(
            {
                f"label_{jet_type}": np.ones(len(x_reco_ak)) * (jet_type_name == jet_type)
                for jet_type in jet_types_dict
            }
        )

        return p4s_reco, x_reco_ak, labels_onehot

    return p4s_reco, x_reco_ak
