"""Callback for evaluating the classifier."""

import os
from typing import Callable

import awkward as ak
import lightning as L

# import cplt
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf
from sklearn.metrics import roc_auc_score

from gabbro.metrics.utils import calc_accuracy, calc_rejection
from gabbro.plotting.utils import decorate_ax, get_col, set_mpl_style
from gabbro.utils.pylogger import get_pylogger

pylogger = get_pylogger("ClassifierEvaluationCallback")


class ClassifierEvaluationCallback(L.Callback):
    def __init__(
        self,
        every_n_epochs: int | Callable = 10,
        additional_eval_epochs: list[int] = None,
        image_path: str = None,
        log_times: bool = True,
        log_epoch_zero: bool = False,
    ):
        super().__init__()
        self.comet_logger = None
        self.image_path = image_path

    def setup_logger(self, rank: int = None) -> None:
        self.logger = get_pylogger(f"{__name__}-ClassifierEvaluationCallback", rank=rank)

    def on_validation_epoch_end(self, trainer, pl_module):
        self.setup_logger(rank=trainer.global_rank)
        self.compare_val_test_vs_train(trainer, pl_module, "val")

    def on_test_end(self, trainer, pl_module):
        if trainer.global_rank != 0:
            return {}

        # save the test predictions
        self.setup_logger(rank=trainer.global_rank)
        save_dir = (
            trainer.default_root_dir + "/plots/" if self.image_path is None else self.image_path
        )
        os.makedirs(save_dir, exist_ok=True)
        save_filename = save_dir + "/test_predictions.parquet"
        self.logger.info(f"Saving test predictions as parquet file to: {save_filename}")
        results_akarr = ak.Array(
            {
                "test_preds": pl_module.test_preds,
                "test_labels": pl_module.test_labels,
            }
        )
        ak.to_parquet(results_akarr, save_filename)

        # calculate the metrics
        results_metrics_dict = self.compare_val_test_vs_train(trainer, pl_module, "test")

        # save the results metrics dict as yaml
        save_filename = save_dir + "/test_metrics.yaml"
        self.logger.info(f"Saving test metrics as yaml file to: {save_filename}")
        OmegaConf.save(results_metrics_dict, save_filename)

        return results_metrics_dict

    def compare_val_test_vs_train(self, trainer, pl_module, stage="val"):
        self.logger.warning("Calling compare_val_test_vs_train")

        if stage == "val":
            if not hasattr(pl_module, "val_preds_list") or not hasattr(
                pl_module, "val_labels_list"
            ):
                self.logger.info("No validation predictions found. Skipping plotting.")
                return
            if not hasattr(pl_module, "train_preds_list") or not hasattr(
                pl_module, "train_labels_list"
            ):
                self.logger.info("No training predictions found. Skipping plotting.")
                return
        elif stage == "test":
            if not hasattr(pl_module, "test_preds_list") or not hasattr(
                pl_module, "test_labels_list"
            ):
                self.logger.info("No test predictions found. Skipping plotting.")
                return

        self.logger.info(
            f"Running ClassifierEvaluationCallback epoch: {trainer.current_epoch} step:"
            f" {trainer.global_step}"
        )
        # get loggers
        for logger in trainer.loggers:
            if isinstance(logger, L.pytorch.loggers.CometLogger):
                self.comet_logger = logger.experiment
                self.logger.info("Found comet logger.")
            elif isinstance(logger, L.pytorch.loggers.WandbLogger):
                self.wandb_logger = logger.experiment
                self.logger.info("Found wandb logger.")

        plot_dir = (
            trainer.default_root_dir + "/plots/" if self.image_path is None else self.image_path
        )
        os.makedirs(plot_dir, exist_ok=True)
        plot_filename = (
            f"{plot_dir}/classifier_output_epoch{trainer.current_epoch}_step{trainer.global_step}.png"
            if stage == "val"
            else f"{plot_dir}/classifier_output_test.pdf"
        )

        if stage == "val":
            n_classes = pl_module.val_labels_list[0].shape[1]
        elif stage == "test":
            n_classes = pl_module.test_labels_list[0].shape[1]

        results_dict_all_classes = {}

        # plot the classifier output
        # calculate the R30/50 values for each signal class (1-9), class 0 is the background
        if stage == "val":
            self.logger.info("Calculating metrics for validation set.")
            pl_module.val_preds = np.concatenate(pl_module.val_preds_list)
            pl_module.val_labels = np.concatenate(pl_module.val_labels_list)
            pl_module.train_preds = np.concatenate(pl_module.train_preds_list)
            pl_module.train_labels = np.concatenate(pl_module.train_labels_list)
            val_is_bkg = pl_module.val_labels[:, 0] == 1
            val_scoreB = pl_module.val_preds[:, 0]
            train_is_bkg = pl_module.train_labels[:, 0] == 1
            train_scoreB = pl_module.train_preds[:, 0]
            if np.sum(val_is_bkg) == 0 or np.sum(train_is_bkg) == 0:
                print("No background jets in this iteration. --> skipping plots.")
                return None
            # calculate multi-class accuracy
            val_acc_multiclass = calc_accuracy(pl_module.val_preds, pl_module.val_labels)
            train_acc_multiclass = calc_accuracy(pl_module.train_preds, pl_module.train_labels)
            results_dict_all_classes["multiclass_accuracy_val"] = val_acc_multiclass
            results_dict_all_classes["multiclass_accuracy_train"] = train_acc_multiclass

        elif stage == "test":
            pl_module.test_preds = np.concatenate(pl_module.test_preds_list)
            pl_module.test_labels = np.concatenate(pl_module.test_labels_list)
            val_is_bkg = pl_module.test_labels[:, 0] == 1
            val_scoreB = pl_module.test_preds[:, 0]
            val_acc_multiclass = calc_accuracy(pl_module.test_preds, pl_module.test_labels)
            results_dict_all_classes["multiclass_accuracy_test"] = float(val_acc_multiclass)

        for sig_class in range(1, n_classes):
            if stage == "val":
                val_is_signal = pl_module.val_labels[:, sig_class] == 1
                train_is_signal = pl_module.train_labels[:, sig_class] == 1
                if np.sum(val_is_signal) == 0 or np.sum(train_is_signal) == 0:
                    print(
                        f"No signal jets for class {sig_class} in this iteration. --> skipping plots."
                    )
                    continue
            elif stage == "test":
                val_is_signal = pl_module.test_labels[:, sig_class] == 1
                if np.sum(val_is_signal) == 0:
                    print(
                        f"No signal jets for class {sig_class} in this iteration. --> skipping plots."
                    )
                    continue

            # get the signal and bkg scores
            val_scoreS = (
                pl_module.val_preds[:, sig_class]
                if stage == "val"
                else pl_module.test_preds[:, sig_class]
            )
            # calculate the score score_S/(score_S + score_B) (like in ParT paper)
            val_scores = val_scoreS / (val_scoreS + val_scoreB)
            if stage == "val":
                train_scoreS = pl_module.train_preds[:, sig_class]
                train_scores = train_scoreS / (train_scoreS + train_scoreB)

            # select only bkg and signal events of the current class
            val_scores = val_scores[val_is_bkg | val_is_signal]
            val_is_signal = val_is_signal[val_is_bkg | val_is_signal]
            if stage == "val":
                train_scores = train_scores[train_is_bkg | train_is_signal]
                train_is_signal = train_is_signal[train_is_bkg | train_is_signal]

            plot_filename_this_class = plot_filename.replace(
                "classifier_output", f"classifier_output_class_{sig_class}"
            )
            # plot the classifier output
            results_dict_all_classes[f"class{sig_class}"] = self.plot_classifier_output(
                val_scores=val_scores,
                train_scores=train_scores if stage == "val" else None,
                val_is_signal=val_is_signal,
                train_is_signal=train_is_signal if stage == "val" else None,
                plot_filename=plot_filename_this_class,
                postfix=f"_class{sig_class}",
                stage=stage,
                trainer=trainer,
            )

        self.logger.info(f"Results for {stage}: {results_dict_all_classes}")

        return results_dict_all_classes

    def plot_classifier_output(
        self,
        val_scores,
        train_scores,
        val_is_signal,
        train_is_signal,
        plot_filename,
        trainer,
        postfix="",
        stage="val",
    ):
        # drop inf and nan values, but notify how many were removed

        if train_scores is not None:
            mask_train_scores_inf_or_nan = np.isnan(train_scores) | np.isinf(train_scores)
            mask_val_scores_inf_or_nan = np.isnan(val_scores) | np.isinf(val_scores)
            n_nan_or_inf_train = np.sum(mask_train_scores_inf_or_nan)
            n_nan_or_inf_val = np.sum(mask_val_scores_inf_or_nan)
        else:
            mask_val_scores_inf_or_nan = np.isnan(val_scores) | np.isinf(val_scores)
            n_nan_or_inf_val = np.sum(mask_val_scores_inf_or_nan)
            n_nan_or_inf_train = 0

        # report how many there are
        if n_nan_or_inf_train != 0:
            self.logger.info(f"Found {n_nan_or_inf_train} nan or inf values in train scores.")
        if n_nan_or_inf_val != 0:
            self.logger.info(f"Found {n_nan_or_inf_val} nan or inf values in val scores.")

        if train_scores is not None and train_is_signal is not None:
            train_scores = train_scores[~mask_train_scores_inf_or_nan]
            train_is_signal = train_is_signal[~mask_train_scores_inf_or_nan]
            train_auc = roc_auc_score(train_is_signal, train_scores)
            r30_train, _ = calc_rejection(scores=train_scores, labels=train_is_signal, sig_eff=0.3)
            r50_train, _ = calc_rejection(scores=train_scores, labels=train_is_signal, sig_eff=0.5)
        else:
            train_auc = 0
            r30_train = 0
            r50_train = 0

        val_scores = val_scores[~mask_val_scores_inf_or_nan]
        val_is_signal = val_is_signal[~mask_val_scores_inf_or_nan]
        val_auc = roc_auc_score(val_is_signal, val_scores)

        r30_val, _ = calc_rejection(scores=val_scores, labels=val_is_signal, sig_eff=0.3)
        r50_val, _ = calc_rejection(scores=val_scores, labels=val_is_signal, sig_eff=0.5)

        # log the r30 values to comet
        if self.comet_logger is not None and stage == "val":
            self.log(f"R30_{stage}{postfix}", r30_val, sync_dist=True, prog_bar=True)
            self.log(f"R50_{stage}{postfix}", r50_val, sync_dist=True, prog_bar=True)
            self.log(f"AUC_{stage}{postfix}", val_auc, sync_dist=True, prog_bar=True)
            if train_scores is not None:
                self.log(f"R30_train{postfix}", r30_train, sync_dist=True, prog_bar=True)
                self.log(f"R50_train{postfix}", r50_train, sync_dist=True, prog_bar=True)
                self.log(f"AUC_train{postfix}", train_auc, sync_dist=True, prog_bar=True)
        # Save the plots
        set_mpl_style()
        fig, ax = plt.subplots(figsize=(5.3, 3))
        hist_kwargs = dict(bins=np.linspace(0, 1, 50), density=True)

        ax.hist(
            val_scores[val_is_signal],
            label=f"Sig {stage}",
            **hist_kwargs,
            color=get_col(0),
            alpha=0.5,
        )
        ax.hist(
            val_scores[~val_is_signal],
            label=f"Bkg {stage}",
            **hist_kwargs,
            color=get_col(1),
            alpha=0.5,
        )
        if train_scores is not None:
            ax.hist(
                train_scores[train_is_signal],
                label="Sig train",
                **hist_kwargs,
                color=get_col(0),
                histtype="step",
            )
            ax.hist(
                train_scores[~train_is_signal],
                label="Bkg train",
                **hist_kwargs,
                color=get_col(1),
                histtype="step",
            )
        ax.set_xlabel("$\\mathrm{score}_{S\\mathrm{vs}B}$")
        ax.set_ylabel("Normalized")
        ax.legend(frameon=False, loc="upper right", ncol=2)
        ax.set_yscale("log")
        decorate_ax(
            ax,
            text=(
                # "Metric: val | train\n"
                f"AUC: {val_auc:.3f} | {train_auc:.3f}\n"
                f"R30: {r30_val:.0f} | {r30_train:.0f}\n"
                f"R50: {r50_val:.0f} | {r50_train:.0f}\n"
            ),
            text_font_size=10,
            top_distance=1.5,
            yscale=1.4,
        )
        fig.tight_layout()
        plt.show()
        # only save and
        if trainer.global_rank == 0:
            self.logger.info(f"Saving plot to: {plot_filename} and upload to comet.")
            fig.savefig(plot_filename, dpi=300)
            if self.comet_logger is not None:
                self.logger.info(f"Logging plot to comet: {plot_filename}")
                self.comet_logger.log_image(plot_filename, name=plot_filename.split("/")[-1])  # noqa: E501
        plt.close()
        results_dict = {
            f"{stage}_auc": float(np.round(val_auc, 5)),
            f"{stage}_r30": float(np.round(r30_val, 2)),
            f"{stage}_r50": float(np.round(r50_val, 2)),
            "train_auc": float(np.round(train_auc, 5)),
            "train_r30": float(np.round(r30_train, 2)),
            "train_r50": float(np.round(r50_train, 2)),
        }
        return results_dict
