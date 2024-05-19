"""Callback for evaluating the generative token model."""

import os
from pathlib import Path

import awkward as ak
import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import vector

import gabbro.plotting.utils as plot_utils
from gabbro.metrics.jet_substructure import JetSubstructure
from gabbro.metrics.utils import calc_quantiled_kl_divergence_for_dict
from gabbro.plotting.feature_plotting import plot_features
from gabbro.utils.arrays import np_to_ak
from gabbro.utils.pylogger import get_pylogger

pylogger = get_pylogger("GenEvalCallback")
vector.register_awkward()


class GenEvalCallback(L.Callback):
    def __init__(
        self,
        image_path: str = None,
        image_filetype: str = "png",
        no_trainer_info_in_filename: bool = False,
        save_result_arrays: bool = None,
        n_val_gen_jets: int = 10,
        starting_at_epoch: int = 0,
        every_n_epochs: int = 1,
        batch_size_for_generation: int = 512,
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
        n_val_gen_jets : int
            Number of validation jets to generate. Default is 10.
        starting_at_epoch : int
            Start evaluating the model at this epoch. Default is 0.
        every_n_epochs : int
            Evaluate the model every n epochs. Default is 1.
        batch_size_for_generation : int
            Batch size for generating the jets. Default is 512.
        """
        super().__init__()
        self.comet_logger = None
        self.image_path = image_path
        self.n_val_gen_jets = n_val_gen_jets
        self.image_filetype = image_filetype
        self.no_trainer_info_in_filename = no_trainer_info_in_filename
        self.save_results_arrays = save_result_arrays
        self.every_n_epochs = every_n_epochs
        self.starting_at_epoch = starting_at_epoch
        self.batch_size_for_generation = batch_size_for_generation

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch < self.starting_at_epoch:
            pylogger.info(
                "Skipping generation. Starting evaluating with this callback"
                f" at epoch {self.starting_at_epoch}."
            )
            return None
        if trainer.current_epoch % self.every_n_epochs != 0:
            pylogger.info(
                f"Skipping generation. Only evaluating every {self.every_n_epochs} epochs."
            )
            return None
        if len(pl_module.val_token_ids_list) == 0:
            pylogger.warning("No validation data available. Skipping generation.")
            return None
        self.plot_real_vs_gen_jets(trainer, pl_module)

    def on_test_epoch_end(self, trainer, pl_module):
        pass

    def plot_real_vs_gen_jets(self, trainer, pl_module):
        plot_utils.set_mpl_style()

        # get loggers
        for logger in trainer.loggers:
            if isinstance(logger, L.pytorch.loggers.CometLogger):
                self.comet_logger = logger.experiment
            elif isinstance(logger, L.pytorch.loggers.WandbLogger):
                self.wandb_logger = logger.experiment
        # convert the numpy arrays and masks of the real jets to ak arrays of token
        # ids
        np_real_token_ids = np.concatenate(pl_module.val_token_ids_list)
        np_real_token_masks = np.concatenate(pl_module.val_token_masks_list)
        print(f"np_real_token_ids.shape: {np_real_token_ids.shape}")
        print(f"np_real_token_masks.shape: {np_real_token_masks.shape}")
        real_token_ids = np_to_ak(
            x=np_real_token_ids,
            names=["part_token_id"],
            mask=np_real_token_masks,
        )
        self.real_token_ids = ak.values_astype(real_token_ids["part_token_id"], "int64")
        self.gen_token_ids = pl_module.generate_n_jets_batched(
            self.n_val_gen_jets, batch_size=self.batch_size_for_generation
        )

        pylogger.info(f"real_token_ids: {self.real_token_ids}")
        pylogger.info(f"gen_token_ids: {self.gen_token_ids}")

        print(f"Length of generated jets: {len(self.gen_token_ids)}")
        print(f"Length of real jets: {len(self.real_token_ids)}")

        plot_dir = (
            self.image_path
            if self.image_path is not None
            else trainer.default_root_dir + "/plots/"
        )
        os.makedirs(plot_dir, exist_ok=True)
        filename_real = (
            f"{plot_dir}/epoch{trainer.current_epoch}_gstep{trainer.global_step}_real_jets.parquet"
        )
        filename_gen = filename_real.replace("real_jets", "gen_jets")

        # log min max values of the token ids and of the number of constituents
        multiplicity_real = ak.num(self.real_token_ids)
        multiplicity_gen = ak.num(self.gen_token_ids)
        pylogger.info(
            f"Real jets: min multiplicity: {ak.min(multiplicity_real)}, "
            f"max multiplicity: {ak.max(multiplicity_real)}"
        )
        pylogger.info(
            f"Gen jets: min multiplicity: {ak.min(multiplicity_gen)}, "
            f"max multiplicity: {ak.max(multiplicity_gen)}"
        )
        pylogger.info(
            f"Real jets: min token id: {ak.min(self.real_token_ids)}, "
            f"max token id: {ak.max(self.real_token_ids)}"
        )
        pylogger.info(
            f"Gen jets: min token id: {ak.min(self.gen_token_ids)}, "
            f"max token id: {ak.max(self.gen_token_ids)}"
        )

        # check if there are nan values in the token ids
        if np.sum(np.isnan(ak.flatten(self.real_token_ids))) > 0:
            pylogger.warning("Real token ids contain NaN values.")
        if np.sum(np.isnan(ak.flatten(self.gen_token_ids))) > 0:
            pylogger.warning("Generated token ids contain NaN values.")

        ak.to_parquet(self.real_token_ids, filename_real)
        ak.to_parquet(self.gen_token_ids, filename_gen)
        pylogger.info(f"Real jets saved to {filename_real}")
        pylogger.info(f"Generated jets saved to {filename_gen}")

        def reconstruct_ak_array(
            ak_array_filepath, start_token_included, end_token_included, shift_tokens_by_minus_one
        ):
            token_dir = Path(pl_module.token_dir)
            config_file = token_dir / "config.yaml"
            ckpt_file = token_dir / "model_ckpt.ckpt"
            input_file = ak_array_filepath
            output_file = ak_array_filepath.replace(".parquet", "_reco.parquet")

            REPO_DIR = Path(__file__).resolve().parent.parent.parent
            PYTHON_COMMAND = [
                "python",
                f"{REPO_DIR}/scripts/reconstruct_tokens.py",
                f"--tokens_file={input_file}",
                f"--output_file={output_file}",
                f"--ckpt_file={ckpt_file}",
                f"--config_file={config_file}",
                f"--start_token_included={start_token_included}",
                f"--end_token_included={end_token_included}",
                f"--shift_tokens_by_minus_one={shift_tokens_by_minus_one}",
            ]
            os.system(" ".join(PYTHON_COMMAND))  # nosec

            return output_file

        self.real_reco_file = reconstruct_ak_array(filename_real, 1, 1, 1)
        self.gen_reco_file = reconstruct_ak_array(filename_gen, 1, 0, 1)

        p4s_real = ak.from_parquet(self.real_reco_file)
        p4s_gen = ak.from_parquet(self.gen_reco_file)

        substructure_real = JetSubstructure(p4s_real[ak.num(p4s_real) >= 3])
        substructure_gen = JetSubstructure(p4s_gen[ak.num(p4s_gen) >= 3])

        substructure_real_ak = substructure_real.get_substructure_as_ak_array()
        substructure_gen_ak = substructure_gen.get_substructure_as_ak_array()

        print(f"Plotting {len(p4s_real)} real jets and {len(p4s_gen)} generated jets...")

        names_labels_dict_for_plotting = {
            "jet_pt": "Jet $p_T$ [GeV]",
            "jet_eta": "Jet $\\eta$",
            "jet_phi": "Jet $\\phi$",
            "jet_mass": "Jet mass [GeV]",
            "tau32": "$\\tau_{32}$",
            "tau21": "$\\tau_{21}$",
            "jet_n_constituents": "Number of constituents",
        }

        fig, axarr = plot_features(
            ak_array_dict={
                "Real jets": substructure_real_ak,
                "Gen. jets": substructure_gen_ak,
            },
            names=names_labels_dict_for_plotting,
            bins_dict={
                "jet_pt": np.linspace(450, 1050, 70),
                "jet_eta": np.linspace(-0.1, 0.1, 70),
                "jet_phi": np.linspace(-0.01, 0.01, 70),
                "jet_mass": np.linspace(0, 250, 70),
                "tau32": np.linspace(0, 1.2, 70),
                "tau21": np.linspace(0, 1.2, 70),
                "jet_n_constituents": np.linspace(-0.5, 128.5, 130),
            },
            flatten=False,
            ax_rows=2,
            legend_only_on=0,
        )

        # calculate the kld between the real and generated jets
        kld_dict = calc_quantiled_kl_divergence_for_dict(
            dict_reference=substructure_real_ak,
            dict_approx=substructure_gen_ak,
            names=list(names_labels_dict_for_plotting.keys()),
            n_bins=50,
        )
        pylogger.info(f"KLD values: {kld_dict}")

        image_filename = (
            f"{plot_dir}/epoch{trainer.current_epoch}_gstep{trainer.global_step}_"
            f"kldJetMass_{kld_dict['jet_mass']}_real_vs_gen_jets.{self.image_filetype}"
        )
        fig.savefig(image_filename)

        plt.show()

        if self.comet_logger is not None:
            for key, value in kld_dict.items():
                self.comet_logger.log_metric(f"val_kld_{key}", value, step=trainer.global_step)
            self.comet_logger.log_image(
                image_filename, name=image_filename.split("/")[-1], step=trainer.global_step
            )
