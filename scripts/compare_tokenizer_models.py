# setup logging in notebook
import argparse
import logging
from pathlib import Path

import awkward as ak
import numpy as np
from omegaconf import OmegaConf
from yaml import safe_load

import gabbro.plotting.utils as plot_utils
from gabbro.plotting.feature_plotting import plot_hist_with_ratios
from gabbro.utils.jet_types import jet_types_dict

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO)
logger.info("test")
plot_utils.set_mpl_style()


parser = argparse.ArgumentParser()
parser.add_argument("--config")


def get_results_arrays(cfg):
    print(OmegaConf.to_yaml(cfg))

    array_dict = {
        id: {
            "arr": ak.from_parquet(entry["path"]),
            "hist_kwargs": entry["hist_kwargs"],
        }
        for id, entry in cfg.items()
        if not id.startswith(".")
    }
    return array_dict


def main(cfg_file):
    cfg_path = Path(cfg_file)
    cfg = safe_load(open(cfg_path))
    output_dir = cfg_path.parent
    print(output_dir)
    array_dict = get_results_arrays(cfg)
    # ---------------------------------------------------
    # all-types inclusive plots
    fig_mass_all_types, axes_mass_all_types = plot_utils.get_ax_and_fig_for_ratio_plot(
        figsize=(3.8, 3.8)
    )
    fig_pt_all_types, axes_pt_all_types = plot_utils.get_ax_and_fig_for_ratio_plot(
        figsize=(3.8, 3.5)
    )

    plot_hist_with_ratios(
        xlabel="Jet mass [GeV]",
        ax_upper=axes_mass_all_types[0],
        ax_ratio=axes_mass_all_types[1],
        ref_dict={
            "Original": {
                "arr": list(array_dict.values())[0]["arr"].jet_substructure_original.jet_mass,
                "hist_kwargs": list(array_dict.values())[0]["hist_kwargs"],
            }
        },
        comp_dict={
            id: {
                "arr": arr_dict["arr"].jet_substructure_reco.jet_mass,
                "hist_kwargs": arr_dict["hist_kwargs"],
            }
            for id, arr_dict in array_dict.items()
        },
        ratio_range=[0.55, 1.45],
        bins=np.linspace(0, 300, 80),
        leg_ncols=1,
        leg_loc="upper center",
    )
    # fig_mass_all_types.suptitle("Jet mass comparison", fontsize=15)
    plot_utils.decorate_ax(axes_mass_all_types[0], yscale=2.1, remove_first_ytick=True)
    axes_mass_all_types[1].set_ylabel("Ratio to original")
    fig_mass_all_types.tight_layout()
    fig_mass_all_types.savefig(output_dir / "jet_mass_comparison_all_types.pdf")

    plot_hist_with_ratios(
        xlabel="Jet $p_\\mathrm{T}$ [GeV]",
        ax_upper=axes_pt_all_types[0],
        ax_ratio=axes_pt_all_types[1],
        ref_dict={
            "Original": {
                "arr": list(array_dict.values())[0]["arr"].jet_p4s_original.pt,
                "hist_kwargs": list(array_dict.values())[0]["hist_kwargs"],
            }
        },
        comp_dict={
            id: {
                "arr": arr_dict["arr"].jet_p4s_reco.pt,
                "hist_kwargs": arr_dict["hist_kwargs"],
            }
            for id, arr_dict in array_dict.items()
        },
        ratio_range=[0.55, 1.45],
        bins=np.linspace(450, 1050, 61),
        leg_ncols=1,
        leg_loc="upper center",
    )
    plot_utils.decorate_ax(axes_pt_all_types[0], yscale=1.5, remove_first_ytick=True)
    axes_pt_all_types[1].set_ylabel("Ratio to\noriginal")
    fig_pt_all_types.tight_layout()
    fig_pt_all_types.savefig(output_dir / "jet_pt_comparison_all_types.pdf")

    # ---------------------------------------------------
    # per-type plots

    figsize_res = (20, 7.5)
    figsize_ratio = (20, 7.5)
    leg_cols = 1
    # ---------- particle pt, eta, phi resolution ----------
    fig_pt_res, axes_pt_res = plot_utils.get_ax_and_fig_2x5(ratio=False, figsize=figsize_res)
    fig_eta_res, axes_eta_res = plot_utils.get_ax_and_fig_2x5(ratio=False, figsize=figsize_res)
    fig_phi_res, axes_phi_res = plot_utils.get_ax_and_fig_2x5(ratio=False, figsize=figsize_res)
    # ---------- jet mass ----------
    # resolution
    fig_jet_mass_res, axes_jet_mass_res = plot_utils.get_ax_and_fig_2x5(
        ratio=False, figsize=figsize_res
    )
    # original vs reco
    fig_jet_mass_comp, axes_jet_mass_comp = plot_utils.get_ax_and_fig_2x5(
        ratio=True, figsize=figsize_ratio
    )
    # ------------ jet pT ------------
    # resolution
    fig_jet_pt_res, axes_jet_pt_res = plot_utils.get_ax_and_fig_2x5(
        ratio=False, figsize=figsize_res
    )
    # original vs reco
    fig_jet_pt_comp, axes_jet_pt_comp = plot_utils.get_ax_and_fig_2x5(
        ratio=True, figsize=figsize_ratio
    )

    # ---------- tau32 ----------
    # resolution
    fig_tau32_res, axes_tau32_res = plot_utils.get_ax_and_fig_2x5(ratio=False, figsize=figsize_res)
    # original vs reco
    fig_tau32_comp, axes_tau32_comp = plot_utils.get_ax_and_fig_2x5(
        ratio=True, figsize=figsize_ratio
    )
    # ---------- tau21 ----------
    # resolution
    fig_tau21_res, axes_tau21_res = plot_utils.get_ax_and_fig_2x5(ratio=False, figsize=figsize_res)
    # original vs reco
    fig_tau21_comp, axes_tau21_comp = plot_utils.get_ax_and_fig_2x5(
        ratio=True, figsize=figsize_ratio
    )

    for i, (jet_type, jet_type_dict) in enumerate(jet_types_dict.items()):
        jet_type_mask = list(array_dict.values())[0]["arr"].labels == jet_type_dict["label"]

        # ---------- particle pt, eta, phi resolution ----------
        plot_hist_with_ratios(
            xlabel="Particle $p_\\mathrm{T}^\\mathrm{reco} - p_\\mathrm{T}^\\mathrm{original}$ [GeV]",
            # xlabel="Particle $\\frac{p_T^\\text{reco} - p_T^\\text{original}}{p_T^\\text{original}}$",
            ax_upper=axes_pt_res[i],
            comp_dict={
                id: {
                    "arr": ak.flatten(arr_dict["arr"].part_p4s_reco[jet_type_mask].pt)
                    - ak.flatten(arr_dict["arr"].part_p4s_original[jet_type_mask].pt),
                    # "arr": (ak.flatten(arr_dict["arr"].part_p4s_reco[jet_type_mask].pt)
                    # - ak.flatten(arr_dict["arr"].part_p4s_original[jet_type_mask].pt)) / ak.flatten(arr_dict["arr"].part_p4s_original[jet_type_mask].pt),
                    "hist_kwargs": arr_dict["hist_kwargs"],
                }
                for id, arr_dict in array_dict.items()
            },
            bins=np.linspace(-1, 1, 50),
            # bins=np.linspace(-0.1, 0.1, 50),
            leg_ncols=leg_cols,
            leg_loc="upper center",
        )
        plot_hist_with_ratios(
            xlabel="Particle $\\eta_\\mathrm{rel}^\\mathrm{reco} - \\eta_\\mathrm{rel}^\\mathrm{original}$",
            ax_upper=axes_eta_res[i],
            comp_dict={
                id: {
                    "arr": ak.flatten(arr_dict["arr"].part_p4s_reco[jet_type_mask].eta)
                    - ak.flatten(arr_dict["arr"].part_p4s_original[jet_type_mask].eta),
                    "hist_kwargs": arr_dict["hist_kwargs"],
                }
                for id, arr_dict in array_dict.items()
            },
            bins=np.linspace(-0.05, 0.05, 50),
            leg_ncols=leg_cols,
            leg_loc="upper center",
        )
        plot_hist_with_ratios(
            xlabel="Particle $\\phi_\\mathrm{rel}^\\mathrm{reco} - \\phi_\\mathrm{rel}^\\mathrm{original}$",
            ax_upper=axes_phi_res[i],
            comp_dict={
                id: {
                    "arr": ak.flatten(arr_dict["arr"].part_p4s_reco[jet_type_mask].phi)
                    - ak.flatten(arr_dict["arr"].part_p4s_original[jet_type_mask].phi),
                    "hist_kwargs": arr_dict["hist_kwargs"],
                }
                for id, arr_dict in array_dict.items()
            },
            bins=np.linspace(-0.05, 0.05, 50),
            leg_ncols=leg_cols,
            leg_loc="upper center",
        )

        # ---------- jet mass ----------
        plot_hist_with_ratios(
            xlabel="Jet $m^\\mathrm{reco} - m^\\mathrm{original}$ [GeV]",
            ax_upper=axes_jet_mass_res[i],
            comp_dict={
                id: {
                    "arr": arr_dict["arr"].jet_substructure_reco.jet_mass[jet_type_mask]
                    - arr_dict["arr"].jet_substructure_original.jet_mass[jet_type_mask],
                    "hist_kwargs": arr_dict["hist_kwargs"],
                }
                for id, arr_dict in array_dict.items()
            },
            # bins=np.linspace(-15, 15, 70),
            bins=np.linspace(-70, 30, 70),
            leg_ncols=leg_cols,
            leg_loc="upper center",
        )
        plot_hist_with_ratios(
            xlabel="Jet mass [GeV]",
            ax_upper=axes_jet_mass_comp[0, i],
            ax_ratio=axes_jet_mass_comp[1, i],
            ref_dict={
                "Original": {
                    "arr": list(array_dict.values())[0]["arr"].jet_substructure_original.jet_mass[
                        jet_type_mask
                    ],
                    "hist_kwargs": list(array_dict.values())[0]["hist_kwargs"],
                }
            },
            comp_dict={
                id: {
                    "arr": arr_dict["arr"].jet_substructure_reco.jet_mass[jet_type_mask],
                    "hist_kwargs": arr_dict["hist_kwargs"],
                }
                for id, arr_dict in array_dict.items()
            },
            ratio_range=[0.7, 1.3],
            bins=np.linspace(0, 250, 50),
            leg_ncols=leg_cols,
            leg_loc="upper center",
        )

        # ------------ jet pT ------------
        plot_hist_with_ratios(
            xlabel="Jet $p_\\mathrm{T}^\\mathrm{reco} - p_\\mathrm{T}^\\mathrm{original}$ [GeV]",
            ax_upper=axes_jet_pt_res[i],
            comp_dict={
                id: {
                    "arr": arr_dict["arr"].jet_p4s_reco.pt[jet_type_mask]
                    - arr_dict["arr"].jet_p4s_original.pt[jet_type_mask],
                    "hist_kwargs": arr_dict["hist_kwargs"],
                }
                for id, arr_dict in array_dict.items()
            },
            bins=np.linspace(-80, 80, 70),
            leg_ncols=leg_cols,
            leg_loc="upper center",
        )
        plot_hist_with_ratios(
            xlabel="Jet $p_\\mathrm{T}$ [GeV]",
            ax_upper=axes_jet_pt_comp[0, i],
            ax_ratio=axes_jet_pt_comp[1, i],
            ref_dict={
                "Original": {
                    "arr": list(array_dict.values())[0]["arr"].jet_p4s_original.pt[jet_type_mask],
                    "hist_kwargs": list(array_dict.values())[0]["hist_kwargs"],
                }
            },
            comp_dict={
                id: {
                    "arr": arr_dict["arr"].jet_p4s_reco.pt[jet_type_mask],
                    "hist_kwargs": arr_dict["hist_kwargs"],
                }
                for id, arr_dict in array_dict.items()
            },
            ratio_range=[0.7, 1.3],
            bins=np.linspace(300, 1050, 61),
            leg_ncols=leg_cols,
            leg_loc="upper center",
        )

        # # ---------- jet tau32 ----------
        plot_hist_with_ratios(
            xlabel="Jet $\\tau_{32}^\\mathrm{reco} - \\tau_{32}^\\mathrm{original}$",
            ax_upper=axes_tau32_res[i],
            comp_dict={
                id: {
                    "arr": arr_dict["arr"].jet_substructure_reco.tau32[jet_type_mask]
                    - arr_dict["arr"].jet_substructure_original.tau32[jet_type_mask],
                    "hist_kwargs": arr_dict["hist_kwargs"],
                }
                for id, arr_dict in array_dict.items()
            },
            bins=np.linspace(-0.4, 0.4, 70),
            leg_ncols=leg_cols,
            leg_loc="upper center",
        )

        plot_hist_with_ratios(
            xlabel="Jet $\\tau_{{32}}$",
            ax_upper=axes_tau32_comp[0, i],
            ax_ratio=axes_tau32_comp[1, i],
            ref_dict={
                "Original": {
                    "arr": list(array_dict.values())[0]["arr"].jet_substructure_original.tau32[
                        jet_type_mask
                    ],
                    "hist_kwargs": list(array_dict.values())[0]["hist_kwargs"],
                }
            },
            comp_dict={
                id: {
                    "arr": arr_dict["arr"].jet_substructure_reco.tau32[jet_type_mask],
                    "hist_kwargs": arr_dict["hist_kwargs"],
                }
                for id, arr_dict in array_dict.items()
            },
            ratio_range=[0.7, 1.3],
            bins=np.linspace(0, 1.2, 50),
            leg_ncols=leg_cols,
            leg_loc="upper center",
        )

        # # ---------- jet tau21 ----------
        plot_hist_with_ratios(
            xlabel="Jet $\\tau_{21}^\\mathrm{reco} - \\tau_{21}^\\mathrm{original}$",
            ax_upper=axes_tau21_res[i],
            comp_dict={
                id: {
                    "arr": arr_dict["arr"].jet_substructure_reco.tau21[jet_type_mask]
                    - arr_dict["arr"].jet_substructure_original.tau21[jet_type_mask],
                    "hist_kwargs": arr_dict["hist_kwargs"],
                }
                for id, arr_dict in array_dict.items()
            },
            bins=np.linspace(-0.4, 0.4, 70),
            leg_ncols=leg_cols,
            leg_loc="upper center",
        )

        plot_hist_with_ratios(
            xlabel="Jet $\\tau_{{21}}$",
            ax_upper=axes_tau21_comp[0, i],
            ax_ratio=axes_tau21_comp[1, i],
            ref_dict={
                "Original": {
                    "arr": list(array_dict.values())[0]["arr"].jet_substructure_original.tau21[
                        jet_type_mask
                    ],
                    "hist_kwargs": list(array_dict.values())[0]["hist_kwargs"],
                }
            },
            comp_dict={
                id: {
                    "arr": arr_dict["arr"].jet_substructure_reco.tau21[jet_type_mask],
                    "hist_kwargs": arr_dict["hist_kwargs"],
                }
                for id, arr_dict in array_dict.items()
            },
            ratio_range=[0.7, 1.3],
            bins=np.linspace(0, 1.2, 50),
            leg_ncols=leg_cols,
            leg_loc="upper center",
        )

        axes_pt_res[i].set_title(jet_type_dict["tex_label"])
        axes_eta_res[i].set_title(jet_type_dict["tex_label"])
        axes_phi_res[i].set_title(jet_type_dict["tex_label"])
        axes_jet_mass_comp[0, i].set_title(jet_type_dict["tex_label"])
        axes_jet_mass_res[i].set_title(jet_type_dict["tex_label"])
        axes_jet_pt_comp[0, i].set_title(jet_type_dict["tex_label"])
        axes_jet_pt_res[i].set_title(jet_type_dict["tex_label"])
        axes_tau32_res[i].set_title(jet_type_dict["tex_label"])
        axes_tau32_comp[0, i].set_title(jet_type_dict["tex_label"])
        axes_tau21_res[i].set_title(jet_type_dict["tex_label"])
        axes_tau21_comp[0, i].set_title(jet_type_dict["tex_label"])

        axes_jet_mass_comp[1, i].set_ylabel("Ratio to original")
        axes_jet_pt_comp[1, i].set_ylabel("Ratio to original")
        axes_tau32_comp[1, i].set_ylabel("Ratio to original")
        axes_tau21_comp[1, i].set_ylabel("Ratio to original")

        zero_line_ymax = 0.6
        vline_kwargs = dict(color="black", linestyle="-", ymax=zero_line_ymax, alpha=0.5)
        axes_pt_res[i].axvline(0, **vline_kwargs)
        axes_eta_res[i].axvline(0, **vline_kwargs)
        axes_phi_res[i].axvline(0, **vline_kwargs)
        axes_jet_mass_res[i].axvline(0, **vline_kwargs)
        axes_jet_pt_res[i].axvline(0, **vline_kwargs)
        axes_tau32_res[i].axvline(0, **vline_kwargs)
        axes_tau21_res[i].axvline(0, **vline_kwargs)

        plot_utils.decorate_ax(axes_jet_mass_comp[0, i], yscale=2.2, remove_first_ytick=True)
        plot_utils.decorate_ax(axes_jet_pt_comp[0, i], yscale=2.2, remove_first_ytick=True)
        plot_utils.decorate_ax(axes_tau32_comp[0, i], yscale=2.2, remove_first_ytick=True)
        plot_utils.decorate_ax(axes_tau21_comp[0, i], yscale=2.2, remove_first_ytick=True)

        yscale_res = 1.7
        plot_utils.decorate_ax(axes_pt_res[i], yscale=yscale_res)
        plot_utils.decorate_ax(axes_eta_res[i], yscale=yscale_res)
        plot_utils.decorate_ax(axes_phi_res[i], yscale=yscale_res)
        plot_utils.decorate_ax(axes_jet_mass_res[i], yscale=yscale_res)
        plot_utils.decorate_ax(axes_jet_pt_res[i], yscale=yscale_res)
        plot_utils.decorate_ax(axes_tau32_res[i], yscale=yscale_res)
        plot_utils.decorate_ax(axes_tau21_res[i], yscale=yscale_res)

    # fig_pt_res.suptitle(
    #     "Particle $p_\\mathrm{T}^\\mathrm{reco} - p_\\mathrm{T}^\\mathrm{original}$ [GeV]",
    #     fontsize=15,
    # )
    # fig_eta_res.suptitle(
    #     "Particle $\\eta_\\mathrm{rel}^\\mathrm{reco} - \\eta_\\mathrm{rel}^\\mathrm{original}$",
    #     fontsize=15,
    # )
    # fig_phi_res.suptitle(
    #     "Particle $\\phi_\\mathrm{rel}^\\mathrm{reco} - \\phi_\\mathrm{rel}^\\mathrm{original}$",
    #     fontsize=15,
    # )
    # fig_jet_mass_res.suptitle("Jet $m^\\mathrm{reco} - m^\\mathrm{original}$", fontsize=15)
    # fig_jet_mass_comp.suptitle("Jet mass comparison", fontsize=15)
    # fig_jet_pt_res.suptitle(
    #     "Jet $p_\\mathrm{T}^\\mathrm{reco} - p_\\mathrm{T}^\\mathrm{original}$ [GeV]", fontsize=15
    # )
    # fig_jet_pt_comp.suptitle("Jet $p_\\mathrm{T}$ comparison", fontsize=15)
    # fig_tau32_res.suptitle(
    #     "Jet $\\tau_{32}^\\mathrm{reco} - \\tau_{32}^\\mathrm{original}$", fontsize=15
    # )
    # fig_tau32_comp.suptitle("Jet $\\tau_{32}$ comparison", fontsize=15)
    # fig_tau21_res.suptitle(
    #     "Jet $\\tau_{21}^\\mathrm{reco} - \\tau_{21}^\\mathrm{original}$", fontsize=15
    # )
    # fig_tau21_comp.suptitle("Jet $\\tau_{21}$ comparison", fontsize=15)
    fig_pt_res.tight_layout()
    fig_eta_res.tight_layout()
    fig_phi_res.tight_layout()
    fig_jet_mass_comp.tight_layout()
    fig_jet_mass_res.tight_layout()
    fig_jet_pt_res.tight_layout()
    fig_jet_pt_comp.tight_layout()
    fig_tau32_res.tight_layout()
    fig_tau32_comp.tight_layout()
    fig_tau21_res.tight_layout()
    fig_tau21_comp.tight_layout()

    # save figures
    print(f"Saving figures to {output_dir}")
    fig_pt_res.savefig(output_dir / "particle_pt_resolution.pdf")
    fig_eta_res.savefig(output_dir / "particle_eta_resolution.pdf")
    fig_phi_res.savefig(output_dir / "particle_phi_resolution.pdf")
    fig_jet_mass_res.savefig(output_dir / "jet_mass_resolution.pdf")
    fig_jet_mass_comp.savefig(output_dir / "jet_mass_comparison.pdf")
    fig_jet_pt_res.savefig(output_dir / "jet_pt_resolution.pdf")
    fig_jet_pt_comp.savefig(output_dir / "jet_pt_comparison.pdf")
    fig_tau32_res.savefig(output_dir / "tau32_resolution.pdf")
    fig_tau32_comp.savefig(output_dir / "tau32_comparison.pdf")
    fig_tau21_res.savefig(output_dir / "tau21_resolution.pdf")
    fig_tau21_comp.savefig(output_dir / "tau21_comparison.pdf")

    # save resolution plots also as subplots
    for i, (jet_type, jet_type_dict) in enumerate(jet_types_dict.items()):
        # remove the title
        axes_jet_mass_res[i].set_title("")
        axes_tau32_res[i].set_title("")
        axes_tau21_res[i].set_title("")

        plot_utils.save_subplot(
            fig=fig_jet_mass_res,
            ax=axes_jet_mass_res[i],
            saveas=output_dir / f"jet_mass_resolution_{jet_type}.pdf",
        )
        plot_utils.save_subplot(
            fig=fig_tau32_res,
            ax=axes_tau32_res[i],
            saveas=output_dir / f"tau32_resolution_{jet_type}.pdf",
        )
        plot_utils.save_subplot(
            fig=fig_tau21_res,
            ax=axes_tau21_res[i],
            saveas=output_dir / f"tau21_resolution_{jet_type}.pdf",
        )


if __name__ == "__main__":
    args = parser.parse_args()
    main(args.config)
