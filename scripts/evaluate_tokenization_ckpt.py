# setup logging in notebook
import argparse
import logging
import sys
from pathlib import Path

import awkward as ak
import hydra
import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf

import gabbro
import gabbro.plotting.utils as plot_utils
from gabbro.callbacks.tokenization_callback import TokenizationEvalCallback
from gabbro.models.vqvae import VQVAELightning
from gabbro.plotting.feature_plotting import plot_features_pairplot
from gabbro.utils.pylogger import get_pylogger

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate tokenization ckpt")
    parser.add_argument(
        "--ckpt_path",
        type=Path,
        help="Path to the checkpoint to evaluate",
    )
    parser.add_argument(
        "--n_eval",
        type=int,
        default=1000,
        help="Number of jets to evaluate per jet type",
    )
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Run in development mode (i.e. with less data)",
        default=False,
    )
    return parser.parse_args()


def main(ckpt_path, n_eval, dev):
    plot_utils.set_mpl_style()
    logger = get_pylogger(__name__)
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    # complete the path
    ckpt_path = list(ckpt_path.parent.glob(ckpt_path.name))[0]
    config_path = ckpt_path.parent.parent / "config.yaml"
    cfg = OmegaConf.load(config_path)
    cfg.data.dataset_kwargs_common["n_jets_per_file"] = n_eval
    logger.info(f"Loading config from: {config_path}")

    if dev:
        tbqq_cfg = cfg.data.dataset_kwargs_test.files_dict["Tbqq"]
        cfg.data.dataset_kwargs_test.files_dict = {"Tbqq": tbqq_cfg}

    out_dir = (
        ckpt_path.parent.parent
        / "evaluated_ckpts"
        / ((ckpt_path.name).split("_loss_")[0] + f"_{n_eval}")
    )
    out_plot_dir = out_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_plot_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {out_dir}")

    # initialize the datamodule and and load the model from checkpoint
    datamodule = hydra.utils.instantiate(cfg.data)
    model = VQVAELightning.load_from_checkpoint(ckpt_path)

    # initialize the trainer to run the test loop
    trainer = L.Trainer(enable_progress_bar=True)
    torch.set_float32_matmul_precision

    # ----------------------------------------------------------
    # --- TEST LOOP ---
    logger.info("----------- Starting test loop -----------")
    trainer.test(model, datamodule=datamodule)
    # trainer.datamodule = datamodule

    callback = TokenizationEvalCallback(
        image_path=out_plot_dir,
        image_filetype="pdf",
        no_trainer_info_in_filename=True,
        save_result_arrays=True,
    )
    callback.on_test_epoch_end(trainer, model)

    pp_dict = cfg.data.dataset_kwargs_common["feature_dict"]
    print(OmegaConf.to_yaml(pp_dict))

    # ----------------------------------------------------------
    # --- Additional plots ---
    # initialize the iterable dataset
    logger.info("----------- Additional plots -----------")
    test_x = datamodule.test_dataset.current_part_data
    test_mask = datamodule.test_dataset.current_part_mask

    model = model.to("cuda")
    model.eval()

    x, mask = test_x, test_mask
    x, mask = x.to("cuda"), mask.to("cuda")
    x_ak = gabbro.utils.arrays.np_to_ak(
        x.detach().cpu().numpy(),
        names=pp_dict.keys(),
        mask=mask.detach().cpu().numpy(),
    )
    x_ak = gabbro.utils.arrays.ak_select_and_preprocess(x_ak, pp_dict, inverse=True)

    p4s_original = ak.zip(
        {
            "pt": x_ak.pt if "pt" in x_ak.fields else x_ak.part_pt,
            "eta": x_ak.etarel if "etarel" in x_ak.fields else x_ak.part_etarel,
            "phi": x_ak.phirel if "phirel" in x_ak.fields else x_ak.part_phirel,
            "mass": ak.zeros_like(x_ak.pt if "pt" in x_ak.fields else x_ak.part_pt),
        },
        with_name="Momentum4D",
    )

    tokens = model.tokenize_ak_array(x_ak, pp_dict, hide_pbar=True)
    tokens_padded, tokens_mask = gabbro.utils.arrays.ak_pad(tokens, 128, return_mask=True)

    model = model.to("cuda")
    model.eval()
    x_reco_ak = model.reconstruct_ak_tokens(tokens, pp_dict, hide_pbar=True)
    p4s_reco = ak.zip(
        {
            "pt": x_reco_ak.pt if "pt" in x_reco_ak.fields else x_reco_ak.part_pt,
            "eta": x_reco_ak.etarel if "etarel" in x_reco_ak.fields else x_reco_ak.part_etarel,
            # "eta": x_ak.etarel if "etarel" in x_ak.fields else x_ak.part_etarel,
            "phi": x_reco_ak.phirel if "phirel" in x_reco_ak.fields else x_reco_ak.part_phirel,
            "mass": ak.zeros_like(x_reco_ak.pt if "pt" in x_reco_ak.fields else x_reco_ak.part_pt),
        },
        with_name="Momentum4D",
    )

    # ----------------------------------------------------------
    # --- PLOT: truth vs reco for each features + resolution ---
    # plot all particle features and their resolution

    logger.info("Plotting truth vs reco for each features + resolution")

    fig, axarr = plt.subplots(2, len(pp_dict.keys()), figsize=(11, 5))
    for i, feat_name in enumerate(pp_dict.keys()):
        ax = axarr[0, i]
        hist_kw = dict(bins=100, alpha=0.8, density=True, histtype="step")
        ax.hist(np.clip(ak.flatten(x_ak[feat_name]), -500, 500), label="original", **hist_kw)
        ax.hist(
            np.clip(ak.flatten(x_reco_ak[feat_name]), -500, 500), label="reconstructed", **hist_kw
        )
        ax.set_xlabel(feat_name)
        ax.legend()
        ax.set_yscale("log")
        ax = axarr[1, i]
        ax.hist(
            np.clip(ak.flatten(x_ak[feat_name]) - ak.flatten(x_reco_ak[feat_name]), -10, 10),
            bins=100,
            label="original",
            alpha=1,
            histtype="step",
            density=True,
        )
        ax.set_xlabel(feat_name + " diff to truth")
    fig.tight_layout()
    fig.set_dpi(300)
    fig.savefig(out_plot_dir / "truth_vs_reco_particle_features.pdf")

    # ----------------------------------------------------------
    # --- PLOT: token perturbation ---

    logger.info("Setting up token perturbation plot...")
    n_tokens = cfg.model.model_kwargs.vq_kwargs["num_codes"]
    perturbed_tokens = []
    n_tokens_to_perturb = n_tokens
    n_perturbations = 500

    random_tokens_to_visualize = np.random.choice(
        n_tokens, size=n_tokens_to_perturb, replace=False
    )

    for i in random_tokens_to_visualize:
        # select random other tokens as "perturbation" - duplicate tokens allowed
        other_tokens = np.random.choice(n_tokens, (n_perturbations, 50), replace=True)
        perturbed_tokens.append(
            ak.Array(
                np.concatenate([np.ones(n_perturbations)[..., None] * i, other_tokens], axis=1)
            )
        )

    perturbed_tokens_recos = []
    for perturbed_token in perturbed_tokens:
        reco = model.reconstruct_ak_tokens(perturbed_token, pp_dict, hide_pbar=True)
        perturbed_tokens_recos.append(reco)

    figs = (2.5, 2.3)
    fig_etaphi, ax_etaphi = plt.subplots(figsize=figs)
    fig_pteta, ax_pteta = plt.subplots(figsize=figs)
    fig_ptphi, ax_ptphi = plt.subplots(figsize=figs)
    save_kwargs = dict(dpi=300, bbox_inches="tight")

    n_tokens_to_plot = len(perturbed_tokens_recos)

    ax_etaphi.set_xlabel(plot_utils.get_label("part_etarel"), labelpad=3)
    ax_etaphi.set_ylabel(plot_utils.get_label("part_phirel"), labelpad=0)
    ax_pteta.set_xlabel(plot_utils.get_label("part_pt"), labelpad=3)
    ax_pteta.set_ylabel(plot_utils.get_label("part_etarel"), labelpad=0)
    ax_ptphi.set_xlabel(plot_utils.get_label("part_pt"), labelpad=3)
    ax_ptphi.set_ylabel(plot_utils.get_label("part_phirel"), labelpad=0)

    logger.info(f"Plotting {n_tokens_to_plot} perturbed tokens")
    for i_token, reco in enumerate(perturbed_tokens_recos[:n_tokens_to_plot]):
        ax_etaphi.scatter(reco["part_etarel"][:, 0], reco["part_phirel"][:, 0], s=1, alpha=0.2)
        ax_pteta.scatter(reco["part_pt"][:, 0], reco["part_etarel"][:, 0], s=1, alpha=0.2)
        ax_ptphi.scatter(reco["part_pt"][:, 0], reco["part_phirel"][:, 0], s=1, alpha=0.2)
        if i_token in [0, 2, 5, 10, 100, n_tokens_to_plot - 1]:
            etalim = 0.9
            philim = 0.9
            ptlim = 230
            ax_etaphi.set_xlim(-etalim, etalim)
            ax_etaphi.set_ylim(-philim, philim + 0.15)
            ax_pteta.set_xlim(0, ptlim)
            ax_pteta.set_ylim(-etalim, etalim + 0.15)
            ax_ptphi.set_xlim(0, ptlim)
            ax_ptphi.set_ylim(-philim, philim + 0.15)
            for ax_ in [ax_etaphi, ax_pteta, ax_ptphi]:
                ax_.tick_params(axis="x", labelsize=8)
                ax_.tick_params(axis="both", which="major", pad=1)
                ax_.tick_params(axis="y", labelsize=8)
            fig_etaphi.tight_layout()
            fig_pteta.tight_layout()
            fig_ptphi.tight_layout()
            fig_etaphi.set_dpi(300)
            fig_pteta.set_dpi(300)
            fig_ptphi.set_dpi(300)
            fig_etaphi.savefig(
                out_plot_dir / f"perturbed_tokens_reco_{i_token + 1}tokens_etaphi.png",
                **save_kwargs,
            )
            fig_pteta.savefig(
                out_plot_dir / f"perturbed_tokens_reco_{i_token + 1}tokens_pteta.png",
                **save_kwargs,
            )
            fig_ptphi.savefig(
                out_plot_dir / f"perturbed_tokens_reco_{i_token + 1}tokens_ptphi.png",
                **save_kwargs,
            )

    # ----------------------------------------------------------
    # --- PLOT: truth vs reco for jet features ---

    logger.info("Plotting jet features")

    fig, axarr = plt.subplots(1, 3, figsize=(7.2, 1.6))
    ax = axarr[0]
    ax.hist(
        ak.num(p4s_original), bins=np.linspace(0, 100, 101), density=True, color="C0", alpha=0.5
    )
    ax.hist(
        ak.num(p4s_reco),
        bins=np.linspace(0, 100, 101),
        histtype="step",
        density=True,
        color="C0",
    )
    ax.set_xlabel("Number of particles per jet")

    jets_original = ak.sum(p4s_original, axis=1)
    jets_reco = ak.sum(p4s_reco, axis=1)

    ax = axarr[1]
    ax.hist(jets_original.pt, bins=60, density=True, color="C0", alpha=0.5, label="Truth")
    ax.hist(jets_reco.pt, bins=60, histtype="step", density=True, color="C0", label="Reco")
    ax.set_xlabel("Jet $p_T$ [GeV]")

    ax = axarr[2]
    bins = np.linspace(0, 250, 70)
    ax.hist(
        np.clip(jets_original.mass, 0, 250),
        bins=bins,
        density=True,
        alpha=0.5,
        color="C0",
        label="Truth",
    )
    ax.hist(
        np.clip(jets_reco.mass, 0, 250),
        bins=bins,
        histtype="step",
        density=True,
        color="C0",
        label="Reco",
    )
    ax.set_xlabel("Jet mass [GeV]")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.set_dpi(300)
    fig.savefig(out_plot_dir / "truth_vs_reco_jet_features.pdf")

    # ----------------------------------------------------------
    # --- PLOT: truth vs reco in eta-phi space (few examples) ---

    logger.info("Plotting truth vs reco in eta-phi space (few examples)")

    fig, axarr = plt.subplots(2, 2, figsize=(6, 5))
    axarr = axarr.flatten()

    for i in range(len(axarr)):
        ax = axarr[i]
        ax.scatter(
            p4s_original[i].eta,
            p4s_original[i].phi,
            s=p4s_original[i].pt,
            label="Truth",
            alpha=0.4,
            color="steelblue",
        )
        ax.scatter(
            p4s_reco[i].eta,
            p4s_reco[i].phi,
            s=p4s_reco[i].pt,
            label="Reco",
            alpha=0.4,
            color="red",
        )
        ax.legend(frameon=True)
        ax.set_xlabel(plot_utils.get_label("part_etarel"))
        ax.set_ylabel(plot_utils.get_label("part_phirel"))
    fig.tight_layout()
    fig.set_dpi(300)
    fig.savefig(out_plot_dir / "truth_vs_reco_examples_scatter.pdf")

    # ----------------------------------------------------------
    # --- PLOT: pairplot of constituents ---

    logger.info("Plotting pairplot of constituent features truth vs. reco")

    array_dict = {
        "pt_true": gabbro.utils.arrays.ak_clip(p4s_original.pt, clip_max=230),
        "eta_true": p4s_original.eta,
        # "rapidity_true": p4s_true.rapidity,
        "phi_true": p4s_original.phi,
        # "mass_true": p4s_true.mass,
        "pt_reco": gabbro.utils.arrays.ak_clip(p4s_reco.pt, clip_max=230),
        # "rapidity_reco": p4s_reco.rapidity,
        "eta_reco": p4s_reco.eta,
        "phi_reco": p4s_reco.phi,
        # "mass_reco": p4s_reco.mass,
    }
    names_dict = {
        "pt_true": "$p_T$ true",
        "eta_true": "$\\eta^\\mathrm{rel}$ true",
        # "rapidity_true": "Rapidity true",
        "phi_true": "$\\phi^\\mathrm{rel}$ true",
        # "mass_true": "mass true",
        "pt_reco": "$p_T$ reco",
        "eta_reco": "$\\eta^\\mathrm{rel}$ reco",
        # "rapidity_reco": "Rapidity reco",
        "phi_reco": "$\\phi^\\mathrm{rel}$ reco",
        # "mass_reco": "mass reco",
    }

    ak_array_to_plot = ak.zip(array_dict)[:10_000]
    g = plot_features_pairplot(
        ak_array_to_plot,
        names=names_dict,
        input_type="ak_constituents",
        pairplot_kwargs={
            "height": 1,
            "grid_kws": {"diag_sharey": False},
            "diag_kws": {"bins": 30, "fill": True},
            "plot_kws": {"bins": 100},
            "corner": True,
        },
    )
    figure = g.fig
    figure.tight_layout()
    figure.align_ylabels()
    figure.align_xlabels()
    # change the tick label size
    for ax in figure.get_axes():
        ax.tick_params(axis="both", labelsize=7)
    # change the spacing between subplots
    figure.subplots_adjust(wspace=0.05, hspace=0.05)
    figure.set_dpi(300)
    figure.savefig(out_plot_dir / "pairplot_constituents.png")


if __name__ == "__main__":
    args = parse_args()
    main(ckpt_path=args.ckpt_path, n_eval=args.n_eval, dev=args.dev)
