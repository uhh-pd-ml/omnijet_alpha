import argparse
import logging
from pathlib import Path

import awkward as ak
import torch

from gabbro.data.data_tokenization import reconstruct_jetclass_file


def main(
    tokens_filename,
    ckpt_file,
    config_file,
    out_file,
    start_token_included=True,
    end_token_included=True,
    shift_tokens_by_minus_one=False,
):
    logger = logging.getLogger(name="reconstruct_tokens")
    logging.basicConfig(level=logging.INFO)

    tokens_filename = Path(tokens_filename)
    ckpt_file = Path(ckpt_file)
    config_file = Path(config_file)

    start_token_included = bool(int(start_token_included))
    end_token_included = bool(int(end_token_included))
    shift_tokens_by_minus_one = bool(int(shift_tokens_by_minus_one))

    gpu_available = torch.cuda.is_available()
    logger.info(f"GPU available: {gpu_available}")
    logger.info(f"Tokens file:   {tokens_filename}")
    logger.info(f"Output file:   {out_file}")
    logger.info(f"Ckpt file:     {ckpt_file}")
    logger.info(f"Config file:   {config_file}")
    logger.info(f"Start token included: {start_token_included}")
    logger.info(f"End token included:   {end_token_included}")
    logger.info(f"Shift tokens by -1:   {shift_tokens_by_minus_one}")

    p4s_reco, x_ak_reco = reconstruct_jetclass_file(
        filename_in=tokens_filename,
        model_ckpt_path=ckpt_file,
        config_path=config_file,
        start_token_included=start_token_included,
        end_token_included=end_token_included,
        shift_tokens_by_minus_one=shift_tokens_by_minus_one,
        device="cuda" if gpu_available else "cpu",
    )
    logger.info(f"Reconstructed p4s: {p4s_reco}")
    logger.info(f"Reconstructed x: {x_ak_reco}")
    logger.info("Saving reconstructed tokens to parquet file.")

    ak.to_parquet(p4s_reco, out_file)
    logger.info(f"Reconstructed tokens saved to {out_file}")
    logger.info("Reconstruction finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens_file", help="Path to the tokens file")
    parser.add_argument(
        "--ckpt_file",
        help="Path to the checkpoint file",
        default="/beegfs/desy/user/birkjosc/datasets/jetclass_tokenized/preliminary_top_qcd/last.ckpt",
        required=False,
    )
    parser.add_argument(
        "--config_file",
        help="Path to the config file",
        default="/beegfs/desy/user/birkjosc/datasets/jetclass_tokenized/preliminary_top_qcd/config.yaml",
        required=False,
    )
    parser.add_argument(
        "--output_file",
        help="Output file name",
        default="/beegfs/desy/user/birkjosc/testing/reconstructed_tokens.parquet",
        required=False,
    )
    parser.add_argument(
        "--start_token_included",
        help="Whether the start token is included in the tokens file",
        default=True,
        required=False,
    )
    parser.add_argument(
        "--end_token_included",
        help="Whether the end token is included in the tokens file",
        default=True,
        required=False,
    )
    parser.add_argument(
        "--shift_tokens_by_minus_one",
        help="Whether to shift the tokens by -1",
        default=False,
        required=False,
    )
    args = parser.parse_args()

    main(
        tokens_filename=args.tokens_file,
        ckpt_file=args.ckpt_file,
        config_file=args.config_file,
        out_file=args.output_file,
        start_token_included=args.start_token_included,
        end_token_included=args.end_token_included,
        shift_tokens_by_minus_one=args.shift_tokens_by_minus_one,
    )
