"""Callback for evaluating the tokenization of particles."""

import os

import awkward as ak
import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import vector

from gabbro.metrics.jet_substructure import JetSubstructure
from gabbro.plotting.feature_plotting import plot_features
from gabbro.utils.arrays import ak_clip, ak_select_and_preprocess, np_to_ak
from gabbro.utils.jet_types import jet_types_dict

# from gabbro.plotting.plotting_functions import plot_p4s
from gabbro.utils.pylogger import get_pylogger

default_labels = {
    "pt": "$p_\\mathrm{T}$",
    "ptrel": "$p_\\mathrm{T}^\\mathrm{rel}$",
    "eta": "$\\eta$",
    "etarel": "$\\eta^\\mathrm{rel}$",
    "phi": "$\\phi$",
    "phirel": "$\\phi^\\mathrm{rel}$",
    "mass": "$m$",
}

pylogger = get_pylogger("TokenizationEvalCallback")
vector.register_awkward()


class TokenizationEvalCallback(L.Callback):
    def __init__(
        self,
        image_path: str = None,
        image_filetype: str = "png",
        no_trainer_info_in_filename: bool = False,
        save_result_arrays: bool = None,
    ):
        """Callback for evaluating the tokenization of particles.

        Parameters
        ----------
        image_path : str
            Path to save the images to. If None, the images are saved to the
            default_root_dir of the trainer.
        image_filetype : str
            Filetype to save the images as. Default is "png".
        no_trainer_info_in_filename : bool
            If True, the filename of the images will not contain the epoch and
            global step information. Default is False.
        save_result_arrays : bool
            If True, the results are saved as parquet file. Default is None.
        """
        super().__init__()
        self.comet_logger = None
        self.image_path = image_path
        self.image_filetype = image_filetype
        self.no_trainer_info_in_filename = no_trainer_info_in_filename
        self.save_results_arrays = save_result_arrays

    def on_validation_epoch_end(self, trainer, pl_module):
        self.plot(trainer, pl_module, stage="val")

    def on_test_epoch_end(self, trainer, pl_module):
        self.plot(trainer, pl_module, stage="test")

    def plot(self, trainer, pl_module, stage="val"):
        if stage == "val" and not hasattr(pl_module, "val_x_original_concat"):
            pylogger.info("No validation predictions found. Skipping plotting.")
            return

        pylogger.info(
            f"Running TokenizationEvalCallback epoch: {trainer.current_epoch} step:"
            f" {trainer.global_step}"
        )
        # get loggers
        for logger in trainer.loggers:
            if isinstance(logger, L.pytorch.loggers.CometLogger):
                self.comet_logger = logger.experiment
            elif isinstance(logger, L.pytorch.loggers.WandbLogger):
                self.wandb_logger = logger.experiment

        plot_dir = (
            self.image_path
            if self.image_path is not None
            else trainer.default_root_dir + "/plots/"
        )
        os.makedirs(plot_dir, exist_ok=True)
        if self.no_trainer_info_in_filename:
            plot_filename = f"{plot_dir}/evaluation_overview.{self.image_filetype}"
        else:
            plot_filename = f"{plot_dir}/epoch{trainer.current_epoch}_gstep{trainer.global_step}_overview.{self.image_filetype}"

        if stage == "val":
            x_recos = pl_module.val_x_reco_concat
            x_originals = pl_module.val_x_original_concat
            masks = pl_module.val_mask_concat
            labels = pl_module.val_labels_concat
            code_idx = pl_module.val_code_idx_concat
        elif stage == "test":
            # return and print that there are no test predictions if there are none
            if not hasattr(pl_module, "test_x_original_concat"):
                pylogger.info("No test predictions found. Skipping plotting.")
                return
            x_recos = pl_module.test_x_reco_concat
            x_originals = pl_module.test_x_original_concat
            masks = pl_module.test_mask_concat
            labels = pl_module.test_labels_concat
            code_idx = pl_module.test_code_idx_concat
        else:
            raise ValueError(f"stage {stage} not recognized")

        if stage == "test":
            pylogger.info(f"x_original_concat.shape: {x_originals.shape}")
            pylogger.info(f"x_reco_concat.shape: {x_recos.shape}")
            pylogger.info(f"masks_concat.shape: {masks.shape}")
            pylogger.info(f"labels_concat.shape: {labels.shape}")

        pp_dict = trainer.datamodule.hparams.dataset_kwargs_common.feature_dict

        # --- only use jets with more than 3 particles (otherwise the calculation
        # of the jet substructure will fail and the plotting will be wrong) ---
        more_than_3_particles_mask = np.sum(masks, axis=1) >= 3
        n_jets_removed = np.sum(~more_than_3_particles_mask)
        if n_jets_removed > 0:
            pylogger.warning(f"Removing {n_jets_removed} jets with less than 3 particles")

        x_recos = x_recos[more_than_3_particles_mask]
        x_originals = x_originals[more_than_3_particles_mask]
        masks = masks[more_than_3_particles_mask]
        labels = labels[more_than_3_particles_mask]
        # ----

        x_reco_ak_pp = np_to_ak(x_recos, mask=masks, names=pp_dict.keys())
        x_original_ak_pp = np_to_ak(x_originals, mask=masks, names=pp_dict.keys())
        x_reco_ak = ak_select_and_preprocess(x_reco_ak_pp, pp_dict=pp_dict, inverse=True)
        x_original_ak = ak_select_and_preprocess(x_original_ak_pp, pp_dict=pp_dict, inverse=True)

        eta_var = None
        phi_var = None
        pt_var = None

        possible_pt_keys = ["part_pt", "pt", "ptrel", "part_ptrel"]
        possible_eta_keys = ["part_eta", "eta", "etarel", "part_etarel"]
        possible_phi_keys = ["part_phi", "phi", "phirel", "part_phirel"]

        for key in possible_eta_keys:
            if key in pp_dict:
                eta_var = key
                break
        for key in possible_pt_keys:
            if key in pp_dict:
                pt_var = key
                break
        for key in possible_phi_keys:
            if key in pp_dict:
                phi_var = key
                break

        p4s_reco_ak = ak.zip(
            {
                "pt": ak_clip(getattr(x_reco_ak, pt_var), clip_min=0.0),
                "eta": getattr(x_reco_ak, eta_var),
                "phi": getattr(x_reco_ak, phi_var),
                "mass": x_reco_ak.mass
                if "mass" in pp_dict
                else ak.zeros_like(getattr(x_reco_ak, pt_var)),
            },
            with_name="Momentum4D",
        )
        p4s_original_ak = ak.zip(
            {
                "pt": getattr(x_original_ak, pt_var),
                "eta": getattr(x_original_ak, eta_var),
                "phi": getattr(x_original_ak, phi_var),
                "mass": x_original_ak.mass
                if "mass" in pp_dict
                else ak.zeros_like(getattr(x_original_ak, pt_var)),
            },
            with_name="Momentum4D",
        )

        p4s_jets_reco_ak = ak.sum(p4s_reco_ak, axis=1)
        p4s_jets_original_ak = ak.sum(p4s_original_ak, axis=1)

        if stage == "val":
            pl_module.val_p4s_reco_ak = p4s_reco_ak
            pl_module.val_p4s_original_ak = p4s_original_ak
        elif stage == "test":
            pl_module.test_p4s_reco_ak = p4s_reco_ak
            pl_module.test_p4s_original_ak = p4s_original_ak
            pl_module.test_p4s_jets_reco_ak = p4s_jets_reco_ak
            pl_module.test_p4s_jets_original_ak = p4s_jets_original_ak
        else:
            raise ValueError(f"stage {stage} not recognized")

        fig, axarr = plot_features(
            ak_array_dict={
                "Truth": p4s_jets_original_ak,
                "Reco": p4s_jets_reco_ak,
            },
            names={feat: default_labels[feat] for feat in ["pt", "eta", "phi", "mass"]},
            label_prefix="Jet",
            flatten=False,
        )
        fig.savefig(plot_filename)

        # make jet mass plot for all jet types
        fig_mass, axarr_mass = plt.subplots(2, 5, figsize=(12, 4))
        fig_massres, axarr_massres = plt.subplots(2, 5, figsize=(12, 4))
        fig_pt, axarr_pt = plt.subplots(2, 5, figsize=(12, 4))
        fig_ptres, axarr_ptres = plt.subplots(2, 5, figsize=(12, 4))
        fig_tau21, axarr_tau21 = plt.subplots(2, 5, figsize=(12, 4))
        fig_tau32, axarr_tau32 = plt.subplots(2, 5, figsize=(12, 4))
        fig_tau21res, axarr_tau21res = plt.subplots(2, 5, figsize=(12, 4))
        fig_tau32res, axarr_tau32res = plt.subplots(2, 5, figsize=(12, 4))
        fig_part_ptres, axarr_part_ptres = plt.subplots(2, 5, figsize=(12, 4))
        fig_part_etares, axarr_part_etares = plt.subplots(2, 5, figsize=(12, 4))
        fig_part_phires, axarr_part_phires = plt.subplots(2, 5, figsize=(12, 4))
        fig_part_massres, axarr_part_massres = plt.subplots(2, 5, figsize=(12, 4))

        axarr_mass = axarr_mass.flatten()
        axarr_massres = axarr_massres.flatten()
        axarr_pt = axarr_pt.flatten()
        axarr_ptres = axarr_ptres.flatten()
        axarr_tau32 = axarr_tau32.flatten()
        axarr_tau21 = axarr_tau21.flatten()
        axarr_tau32res = axarr_tau32res.flatten()
        axarr_tau21res = axarr_tau21res.flatten()
        axarr_part_ptres = axarr_part_ptres.flatten()
        axarr_part_etares = axarr_part_etares.flatten()
        axarr_part_phires = axarr_part_phires.flatten()
        axarr_part_massres = axarr_part_massres.flatten()

        # calculate jet substructure if in test stage (takes some time, so only do it for val stage)
        jet_substructure_original = JetSubstructure(p4s_original_ak)
        jet_substructure_reco = JetSubstructure(p4s_reco_ak)

        # save the results
        if self.image_path is not None and self.save_results_arrays:
            results_ak_array = ak.Array(
                {
                    "part_p4s_reco": p4s_reco_ak,
                    "part_p4s_original": p4s_original_ak,
                    "part_x_reco": x_reco_ak,
                    "part_x_original": x_original_ak,
                    "jet_p4s_reco": p4s_jets_reco_ak,
                    "jet_p4s_original": p4s_jets_original_ak,
                    "jet_substructure_reco": jet_substructure_reco.get_substructure_as_ak_array(),
                    "jet_substructure_original": jet_substructure_original.get_substructure_as_ak_array(),
                    "labels": labels,
                    "masks": masks,
                }
            )
            n_eval_jets = len(p4s_jets_reco_ak)
            out_file_name = f"{self.image_path}/eval_arrays_{n_eval_jets:_}.parquet"
            pylogger.info(f"Saving results to {out_file_name}")
            ak.to_parquet(results_ak_array, out_file_name)

        for i, (jet_type, jet_type_dict) in enumerate(jet_types_dict.items()):
            jet_type_mask = labels == jet_type_dict["label"]

            p4s_jets_reco_ak_i = p4s_jets_reco_ak[jet_type_mask]
            p4s_jets_original_ak_i = p4s_jets_original_ak[jet_type_mask]
            p4s_reco_ak_i = p4s_reco_ak[jet_type_mask]
            p4s_original_ak_i = p4s_original_ak[jet_type_mask]

            hist_kwargs = dict(density=True, color=jet_type_dict["color"])
            hist_kwargs_truth = hist_kwargs | dict(
                alpha=0.4,
                label="Truth",
                histtype="stepfilled",
            )
            hist_kwargs_reco = hist_kwargs | dict(
                label="Reco",
                histtype="step",
            )
            # --------------- Jet-level plots -------------------
            # plot jet substructure

            # plot tau21
            ax = axarr_tau21[i]
            bins = np.linspace(0, 1.1, 70)
            kws_truth = dict(bins=bins) | hist_kwargs_truth
            kws_reco = dict(bins=bins) | hist_kwargs_reco
            # fmt: off
            ax.hist(np.clip(jet_substructure_original.tau21[jet_type_mask], bins[0], bins[-1]), **kws_truth)
            ax.hist(np.clip(jet_substructure_reco.tau21[jet_type_mask], bins[0], bins[-1]), **kws_reco)
            # fmt: on
            ax.set_xlabel("$\\tau_{21}$")
            ax.legend(frameon=False, loc="upper right")
            ax.set_title(jet_type_dict["tex_label"])

            # plot tau32
            ax = axarr_tau32[i]
            bins = np.linspace(0, 1.1, 70)
            kws_truth = dict(bins=bins) | hist_kwargs_truth
            kws_reco = dict(bins=bins) | hist_kwargs_reco
            # fmt: off
            ax.hist(np.clip(jet_substructure_original.tau32[jet_type_mask], bins[0], bins[-1]), **kws_truth)
            ax.hist(np.clip(jet_substructure_reco.tau32[jet_type_mask], bins[0], bins[-1]), **kws_reco)
            # fmt: on
            ax.set_xlabel("$\\tau_{32}$")
            ax.legend(frameon=False, loc="upper right")
            ax.set_title(jet_type_dict["tex_label"])

            # plot tau21 difference between reco and truth
            ax = axarr_tau21res[i]
            bins = np.linspace(-0.5, 0.5, 100)
            ax.hist(
                np.clip(
                    jet_substructure_reco.tau21[jet_type_mask]
                    - jet_substructure_original.tau21[jet_type_mask],
                    bins[0],
                    bins[-1],
                ),
                bins=bins,
                histtype="step",
                **hist_kwargs,
            )
            ax.set_xlabel("$\\tau_{21}$ diff. to truth")
            ax.set_title(jet_type_dict["tex_label"])

            # plot tau32 difference between reco and truth
            ax = axarr_tau32res[i]
            bins = np.linspace(-0.5, 0.5, 100)
            ax.hist(
                np.clip(
                    jet_substructure_reco.tau32[jet_type_mask]
                    - jet_substructure_original.tau32[jet_type_mask],
                    bins[0],
                    bins[-1],
                ),
                bins=bins,
                histtype="step",
                **hist_kwargs,
            )
            ax.set_xlabel("$\\tau_{32}$ diff. to truth")
            ax.set_title(jet_type_dict["tex_label"])

            # plot jet mass
            ax = axarr_mass[i]
            bins = np.linspace(0, 250, 70)
            ax.hist(np.clip(p4s_jets_original_ak_i.mass, 0, 250), bins=bins, **hist_kwargs_truth)
            ax.hist(np.clip(p4s_jets_reco_ak_i.mass, 0, 250), bins=bins, **hist_kwargs_reco)
            ax.set_xlabel("Jet mass [GeV]")
            ax.legend(frameon=False, loc="upper right")
            ax.set_title(jet_type_dict["tex_label"])

            # plot jet mass difference between reco and truth
            ax = axarr_massres[i]
            ax.hist(
                np.clip(p4s_jets_reco_ak_i.mass - p4s_jets_original_ak_i.mass, -50, 50),
                bins=np.linspace(-50, 50, 100),
                histtype="step",
                **hist_kwargs,
            )
            ax.set_xlabel("Jet mass diff. to truth [GeV]")
            ax.set_title(jet_type_dict["tex_label"])

            # plot jet pt
            ax = axarr_pt[i]
            bins = np.linspace(450, 1050, 70)
            ax.hist(np.clip(p4s_jets_original_ak_i.pt, 450, 1050), bins=bins, **hist_kwargs_truth)
            ax.hist(np.clip(p4s_jets_reco_ak_i.pt, 450, 1050), bins=bins, **hist_kwargs_reco)
            ax.set_xlabel("Jet $p_\\mathrm{T}$ [GeV]")
            ax.set_title(jet_type_dict["tex_label"])
            ax.legend(frameon=False, loc="upper right")

            # plot jet pt difference between reco and truth
            ax = axarr_ptres[i]
            bins = np.linspace(-50, 50, 100)
            ax.hist(
                np.clip(p4s_jets_reco_ak_i.pt - p4s_jets_original_ak_i.pt, bins[0], bins[-1]),
                bins=bins,
                histtype="step",
                **hist_kwargs,
            )
            ax.set_xlabel("Jet $p_\\mathrm{T}$ diff. to truth [GeV]")
            ax.set_title(jet_type_dict["tex_label"])

            # --------------- Particle-level plots -------------------
            # plot the particle pt difference between reco and truth
            ax = axarr_part_ptres[i]
            ax.hist(
                np.clip(ak.flatten(p4s_reco_ak_i.pt - p4s_original_ak_i.pt), -5, 5),
                bins=np.linspace(-5, 5, 100),
                histtype="step",
                **hist_kwargs,
            )
            ax.set_xlabel("Particle $p_\\mathrm{T}$ diff. to truth [GeV]")
            ax.set_title(jet_type_dict["tex_label"])

            # plot the particle eta difference between reco and truth
            ax = axarr_part_etares[i]
            ax.hist(
                np.clip(ak.flatten(p4s_reco_ak_i.eta - p4s_original_ak_i.eta), -0.3, 0.3),
                bins=np.linspace(-0.3, 0.3, 100),
                histtype="step",
                **hist_kwargs,
            )
            ax.set_xlabel("Particle $\\eta$ diff. to truth")
            ax.set_title(jet_type_dict["tex_label"])

            # plot the particle phi difference between reco and truth
            ax = axarr_part_phires[i]
            ax.hist(
                np.clip(
                    ak.to_numpy(ak.flatten(p4s_reco_ak_i.phi - p4s_original_ak_i.phi)), -0.3, 0.3
                ),
                bins=np.linspace(-0.3, 0.3, 100),
                histtype="step",
                **hist_kwargs,
            )
            ax.set_title(jet_type_dict["tex_label"])
            ax.set_xlabel("Particle $\\phi$ diff. to truth")

            if "mass" in pp_dict:
                # plot the particle mass difference between reco and truth
                ax = axarr_part_massres[i]
                ax.hist(
                    ak_clip(ak.flatten(p4s_reco_ak_i.mass - p4s_original_ak_i.mass), -5, 5),
                    bins=np.linspace(-5, 5, 100),
                    histtype="step",
                    **hist_kwargs,
                )
                ax.set_xlabel("Particle mass diff. to truth [GeV]")
                ax.set_title(jet_type_dict["tex_label"])

        # fmt: off
        title_kwargs = dict(fontsize=16)
        fig_mass.suptitle("Jet mass", **title_kwargs)
        fig_massres.suptitle("Jet mass difference between reco and truth", **title_kwargs)
        fig_pt.suptitle("Jet $p_\\mathrm{T}$", **title_kwargs)
        fig_ptres.suptitle("Jet $p_\\mathrm{T}$ difference between reco and truth", **title_kwargs)
        fig_tau21.suptitle("Jet $\\tau_{21}$", **title_kwargs)
        fig_tau32.suptitle("Jet $\\tau_{32}$", **title_kwargs)
        fig_tau21res.suptitle("Jet $\\tau_{21}$ difference between reco and truth", **title_kwargs)
        fig_tau32res.suptitle("Jet $\\tau_{32}$ difference between reco and truth", **title_kwargs)
        fig_part_ptres.suptitle("Particle $p_\\mathrm{T}$ difference between reco and truth", **title_kwargs)
        fig_part_etares.suptitle("Particle $\\eta^\\mathrm{rel}$ difference between reco and truth", **title_kwargs)
        fig_part_phires.suptitle("Particle $\\phi^\\mathrm{rel}$ difference between reco and truth", **title_kwargs)
        fig_part_massres.suptitle("Particle mass difference between reco and truth", **title_kwargs)
        # fmt: on

        fig_mass.tight_layout()
        fig_massres.tight_layout()
        fig_pt.tight_layout()
        fig_ptres.tight_layout()
        fig_part_ptres.tight_layout()
        fig_part_etares.tight_layout()
        fig_part_phires.tight_layout()
        fig_part_massres.tight_layout()
        fig_tau21.tight_layout()
        fig_tau32.tight_layout()
        fig_tau21res.tight_layout()
        fig_tau32res.tight_layout()
        rep = "_overview"
        filename_mass_jet_types = plot_filename.replace(rep, "_jet_types_mass")
        filename_massres_jet_types = plot_filename.replace(rep, "_jet_types_massres")
        filename_pt_jet_types = plot_filename.replace(rep, "_jet_types_pt")
        filename_ptres_jet_types = plot_filename.replace(rep, "_jet_types_ptres")
        filename_tau21_jet_types = plot_filename.replace(rep, "_jet_types_tau21")
        filename_tau32_jet_types = plot_filename.replace(rep, "_jet_types_tau32")
        filename_tau21res_jet_types = plot_filename.replace(rep, "_jet_types_tau21res")
        filename_tau32res_jet_types = plot_filename.replace(rep, "_jet_types_tau32res")
        filename_part_ptres_jet_types = plot_filename.replace(rep, "_jet_types_part_ptres")
        filename_part_etares_jet_types = plot_filename.replace(rep, "_jet_types_part_etares")
        filename_part_phires_jet_types = plot_filename.replace(rep, "_jet_types_part_phires")
        filename_part_massres_jet_types = plot_filename.replace(rep, "_jet_types_part_massres")
        fig_mass.savefig(filename_mass_jet_types)
        fig_massres.savefig(filename_massres_jet_types)
        fig_pt.savefig(filename_pt_jet_types)
        fig_ptres.savefig(filename_ptres_jet_types)
        fig_tau32.savefig(filename_tau32_jet_types)
        fig_tau32res.savefig(filename_tau32res_jet_types)
        fig_tau21.savefig(filename_tau21_jet_types)
        fig_tau21res.savefig(filename_tau21res_jet_types)
        fig_part_ptres.savefig(filename_part_ptres_jet_types)
        fig_part_etares.savefig(filename_part_etares_jet_types)
        fig_part_phires.savefig(filename_part_phires_jet_types)
        fig_part_massres.savefig(filename_part_massres_jet_types)

        # log the plots
        if self.comet_logger is not None:
            for fname in [
                plot_filename,
                filename_mass_jet_types,
                filename_massres_jet_types,
                filename_pt_jet_types,
                filename_ptres_jet_types,
                filename_tau21_jet_types,
                filename_tau32_jet_types,
                filename_tau21res_jet_types,
                filename_tau32res_jet_types,
                filename_part_ptres_jet_types,
                filename_part_etares_jet_types,
                filename_part_phires_jet_types,
                filename_part_massres_jet_types,
            ]:
                self.comet_logger.log_image(
                    fname, name=fname.split("/")[-1], step=trainer.global_step
                )

        # calculate the mean abs error of the jet p4s
        abserr_mass = ak.mean(np.abs(p4s_jets_reco_ak.mass - p4s_jets_original_ak.mass))
        abserr_pt = ak.mean(np.abs(p4s_jets_reco_ak.pt - p4s_jets_original_ak.pt))
        abserr_eta = ak.mean(np.abs(p4s_jets_reco_ak.eta - p4s_jets_original_ak.eta))
        abserr_phi = ak.mean(np.abs(p4s_jets_reco_ak.phi - p4s_jets_original_ak.phi))
        abserr_tau21 = np.mean(
            np.abs(jet_substructure_reco.tau21 - jet_substructure_original.tau21)
        )
        abserr_tau32 = np.mean(
            np.abs(jet_substructure_reco.tau32 - jet_substructure_original.tau32)
        )
        # calculate per-feature mean abs error
        shape = x_recos.shape
        x_recos_reshaped = x_recos.reshape(-1, shape[-1])
        x_originals_reshaped = x_originals.reshape(-1, shape[-1])
        particle_feature_mae = np.mean(np.abs(x_recos_reshaped - x_originals_reshaped), axis=1)

        # calculate codebook utilization
        n_codes = pl_module.model.vq_kwargs["num_codes"]
        codebook_utilization = len(np.unique(code_idx)) / n_codes

        # log the mean squared error
        if self.comet_logger is not None:
            self.comet_logger.log_metric("val_abserr_mass", abserr_mass, step=trainer.global_step)
            self.comet_logger.log_metric("val_abserr_pt", abserr_pt, step=trainer.global_step)
            self.comet_logger.log_metric("val_abserr_eta", abserr_eta, step=trainer.global_step)
            self.comet_logger.log_metric("val_abserr_phi", abserr_phi, step=trainer.global_step)
            self.comet_logger.log_metric(
                "val_abserr_tau21", abserr_tau21, step=trainer.global_step
            )
            self.comet_logger.log_metric(
                "val_abserr_tau32", abserr_tau32, step=trainer.global_step
            )
            self.comet_logger.log_metric(
                "val_codebook_utilization", codebook_utilization, step=trainer.global_step
            )
            for i, feature in enumerate(pp_dict.keys()):
                self.comet_logger.log_metric(
                    f"val_abserr_{feature}", particle_feature_mae[i], step=trainer.global_step
                )
