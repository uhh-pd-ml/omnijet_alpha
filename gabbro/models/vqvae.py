import logging
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import vector
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from vqtorch.nn import VectorQuant

from gabbro.utils.arrays import (
    ak_pad,
    ak_select_and_preprocess,
    ak_to_np_stack,
    np_to_ak,
)

vector.register_awkward()

logger = logging.getLogger(__name__)


class VQVAEMLP(torch.nn.Module):
    def __init__(
        self,
        input_dim=2,
        latent_dim=2,
        encoder_layers=None,
        decoder_layers=None,
        vq_kwargs={},
        **kwargs,
    ):
        """Initializes the VQ-VAE model.

        Parameters
        ----------
        codebook_size : int, optional
            The size of the codebook. The default is 8.
        embed_dim : int, optional
            The dimension of the embedding space. The default is 2.
        input_dim : int, optional
            The dimension of the input data. The default is 2.
        encoder_layers : list, optional
            List of integers representing the number of units in each encoder layer.
            If None, a default encoder with a single linear layer is used. The default is None.
        decoder_layers : list, optional
            List of integers representing the number of units in each decoder layer.
            If None, a default decoder with a single linear layer is used. The default is None.
        """

        super().__init__()
        self.vq_kwargs = vq_kwargs
        self.embed_dim = latent_dim
        self.input_dim = input_dim  # for jet constituents, eta and phi

        # --- Encoder --- #
        if encoder_layers is None:
            self.encoder = torch.nn.Linear(self.input_dim, self.embed_dim)
        else:
            enc_layers = []
            enc_layers.append(torch.nn.Linear(self.input_dim, encoder_layers[0]))
            enc_layers.append(torch.nn.ReLU())

            for i in range(len(encoder_layers) - 1):
                enc_layers.append(torch.nn.Linear(encoder_layers[i], encoder_layers[i + 1]))
                enc_layers.append(torch.nn.ReLU())
            enc_layers.append(torch.nn.Linear(encoder_layers[-1], self.embed_dim))

            self.encoder = torch.nn.Sequential(*enc_layers)

        # --- Vector-quantization layer --- #
        self.vqlayer = VectorQuant(feature_size=self.embed_dim, **vq_kwargs)

        # --- Decoder --- #
        if decoder_layers is None:
            self.decoder = torch.nn.Linear(self.embed_dim, self.input_dim)
        else:
            dec_layers = []
            dec_layers.append(torch.nn.Linear(self.embed_dim, decoder_layers[0]))
            dec_layers.append(torch.nn.ReLU())

            for i in range(len(decoder_layers) - 1):
                dec_layers.append(torch.nn.Linear(decoder_layers[i], decoder_layers[i + 1]))
                dec_layers.append(torch.nn.ReLU())
            dec_layers.append(torch.nn.Linear(decoder_layers[-1], self.input_dim))

            self.decoder = torch.nn.Sequential(*dec_layers)

        self.loss_history = []
        self.lr_history = []

    def forward(self, samples, mask=None):
        # mask is there for compatibility with the transformer model
        # encode
        z_embed = self.encoder(samples)
        # quantize
        z_q2, vq_out = self.vqlayer(z_embed)
        # decode
        x_reco = self.decoder(z_q2)
        return x_reco, vq_out


class NormformerBlock(nn.Module):
    def __init__(self, input_dim, mlp_dim, num_heads, dropout_rate=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate

        # define the MultiheadAttention layer with layer normalization
        self.norm1 = nn.LayerNorm(input_dim)
        self.attn = nn.MultiheadAttention(input_dim, num_heads, batch_first=True, dropout=0.1)
        self.norm2 = nn.LayerNorm(input_dim)

        # define the MLP with layer normalization
        self.mlp = nn.Sequential(
            nn.LayerNorm(input_dim),  # Add layer normalization
            nn.Linear(input_dim, mlp_dim),
            nn.SiLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(mlp_dim, input_dim),
        )

        # initialize weights of mlp[-1] and layer norm after attn block to 0
        # such that the residual connection is the identity when the block is
        # initialized
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)
        nn.init.zeros_(self.norm1.weight)

    def forward(self, x, mask=None, return_attn_weights=False):
        # x: (B, S, F)
        # mask: (B, S)
        x = x * mask.unsqueeze(-1)

        # calculate self-attention
        x_norm = self.norm1(x)
        attn_output, attn_weights = self.attn(x_norm, x_norm, x_norm, key_padding_mask=mask != 1)
        # Add residual connection and permute back to (B, S, F)
        attn_res = self.norm2(attn_output) + x

        output = self.mlp(attn_res) + attn_res

        if return_attn_weights:
            return output, attn_weights

        # output shape: (B, S, F)
        return output


class Transformer(torch.nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dim,
        num_heads=1,
        num_blocks=2,
        skip_out_proj=False,
    ):
        super().__init__()

        self.project_in = nn.Linear(input_dim, hidden_dim)

        self.num_blocks = num_blocks
        self.skip_out_proj = skip_out_proj
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.output_dim = output_dim

        self.blocks = nn.ModuleList(
            [
                NormformerBlock(input_dim=hidden_dim, mlp_dim=hidden_dim, num_heads=num_heads)
                for _ in range(num_blocks)
            ]
        )
        self.project_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, mask):
        x = self.project_in(x)
        for i, block in enumerate(self.blocks):
            x = block(x, mask=mask)
        if self.skip_out_proj:
            return x * mask.unsqueeze(-1)
        x = self.project_out(x) * mask.unsqueeze(-1)
        return x


class NormformerStack(torch.nn.Module):
    def __init__(
        self,
        hidden_dim,
        num_heads=1,
        num_blocks=2,
        skip_out_proj=False,
        dropout_rate=0.1,
    ):
        super().__init__()

        self.num_blocks = num_blocks
        self.skip_out_proj = skip_out_proj
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate

        self.blocks = nn.ModuleList(
            [
                NormformerBlock(
                    input_dim=self.hidden_dim,
                    mlp_dim=self.hidden_dim,
                    num_heads=self.num_heads,
                    dropout_rate=self.dropout_rate,
                )
                for _ in range(num_blocks)
            ]
        )

    def forward(self, x, mask):
        for i, block in enumerate(self.blocks):
            x = block(x, mask=mask)
        return x * mask.unsqueeze(-1)


class VQVAETransformer(torch.nn.Module):
    def __init__(
        self,
        input_dim,
        latent_dim,
        hidden_dim,
        num_heads=1,
        num_blocks=2,
        vq_kwargs={},
        **kwargs,
    ):
        super().__init__()

        self.vq_kwargs = vq_kwargs
        self.latent_dim = latent_dim

        self.encoder = Transformer(
            input_dim=input_dim,
            output_dim=latent_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_blocks=num_blocks,
        )
        self.vqlayer = VectorQuant(feature_size=latent_dim, **vq_kwargs)
        self.decoder = Transformer(
            input_dim=latent_dim,
            output_dim=input_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_blocks=num_blocks,
        )
        self.loss_history = []
        self.lr_history = []

    def forward(self, x, mask):
        # encode
        x = self.encoder(x, mask=mask)
        z_embed = x * mask.unsqueeze(-1)
        # quantize
        z, vq_out = self.vqlayer(z_embed)
        # decode
        x_reco = self.decoder(z, mask=mask)
        return x_reco, vq_out


class VQVAENormFormer(torch.nn.Module):
    """This is basically just a re-factor of the VQVAETransformer class, but with more modular
    model components, making it easier to use some components in other models."""

    def __init__(
        self,
        input_dim,
        latent_dim,
        hidden_dim,
        num_heads=1,
        num_blocks=2,
        vq_kwargs={},
        **kwargs,
    ):
        super().__init__()

        self.loss_history = []
        self.lr_history = []

        self.vq_kwargs = vq_kwargs
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_blocks = num_blocks

        # Model components:
        self.input_projection = nn.Linear(self.input_dim, self.hidden_dim)
        self.encoder_normformer = NormformerStack(
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            num_blocks=self.num_blocks,
        )
        self.latent_projection_in = nn.Linear(self.hidden_dim, self.latent_dim)
        self.vqlayer = VectorQuant(feature_size=self.latent_dim, **vq_kwargs)
        self.latent_projection_out = nn.Linear(self.latent_dim, self.hidden_dim)
        self.decoder_normformer = NormformerStack(
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            num_blocks=self.num_blocks,
        )
        self.output_projection = nn.Linear(hidden_dim, input_dim)

    def forward(self, x, mask):
        # encode
        x = self.input_projection(x)
        x = self.encoder_normformer(x, mask=mask)
        z_embed = self.latent_projection_in(x) * mask.unsqueeze(-1)
        # quantize
        z, vq_out = self.vqlayer(z_embed)
        # decode
        x_reco = self.latent_projection_out(z) * mask.unsqueeze(-1)
        x_reco = self.decoder_normformer(x_reco, mask=mask)
        x_reco = self.output_projection(x_reco) * mask.unsqueeze(-1)
        return x_reco, vq_out


class VQVAELightning(L.LightningModule):
    """PyTorch Lightning module for training a VQ-VAE."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler = None,
        model_kwargs={},
        model_type="Transformer",
        **kwargs,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        # --------------- load pretrained model --------------- #
        # if kwargs.get("load_pretrained", False):
        if model_type == "MLP":
            self.model = VQVAEMLP(**model_kwargs)
        elif model_type == "Transformer":
            self.model = VQVAETransformer(**model_kwargs)
        elif model_type == "VQVAENormFormer":
            self.model = VQVAENormFormer(**model_kwargs)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        self.train_loss_history = []
        self.val_loss_list = []

        self.validation_cnt = 0
        self.validation_output = {}

        # loss function (not used atm, since we calc MSE manually)
        self.criterion = torch.nn.MSELoss()

        # for tracking best so far validation accuracy
        self.val_x_original = []
        self.val_x_reco = []
        self.val_mask = []

    def forward(self, x_particle, mask_particle):
        x_particle_reco, vq_out = self.model(x_particle, mask=mask_particle)
        return x_particle_reco, vq_out

    def model_step(self, batch, return_x=False):
        """Perform a single model step on a batch of data."""

        # x_particle, mask_particle, labels = batch
        x_particle = batch["part_features"]
        mask_particle = batch["part_mask"]
        labels = batch["jet_type_labels"]

        x_particle_reco, vq_out = self.forward(x_particle, mask_particle)

        reco_loss = ((x_particle_reco - x_particle) ** 2).mean()
        alpha = self.hparams["model_kwargs"]["alpha"]
        cmt_loss = vq_out["loss"]
        code_idx = vq_out["q"]
        loss = reco_loss + alpha * cmt_loss

        if return_x:
            return loss, x_particle, x_particle_reco, mask_particle, labels, code_idx

        return loss

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set."""
        loss = self.model_step(batch)

        self.train_loss_history.append(float(loss))
        self.log("train_loss", loss.item(), on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def on_train_start(self) -> None:
        self.preprocessing_dict = (
            self.trainer.datamodule.hparams.dataset_kwargs_common.feature_dict
        )

    def on_train_epoch_start(self):
        logger.info(f"Epoch {self.trainer.current_epoch} starting.")
        self.epoch_train_start_time = time.time()  # start timing the epoch

    def on_train_epoch_end(self):
        self.epoch_train_end_time = time.time()
        self.epoch_train_duration_minutes = (
            self.epoch_train_end_time - self.epoch_train_start_time
        ) / 60
        self.log(
            "epoch_train_duration_minutes",
            self.epoch_train_duration_minutes,
            on_epoch=True,
            prog_bar=False,
        )
        logger.info(
            f"Epoch {self.trainer.current_epoch} finished in"
            f" {self.epoch_train_duration_minutes:.1f} minutes."
        )

    def on_train_end(self):
        pass

    def on_validation_epoch_start(self) -> None:
        self.val_x_original = []
        self.val_x_reco = []
        self.val_mask = []
        self.val_labels = []
        self.val_code_idx = []

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        loss, x_original, x_reco, mask, labels, code_idx = self.model_step(batch, return_x=True)

        # save the original and reconstructed data
        self.val_x_original.append(x_original.detach().cpu().numpy())
        self.val_x_reco.append(x_reco.detach().cpu().numpy())
        self.val_mask.append(mask.detach().cpu().numpy())
        self.val_labels.append(labels.detach().cpu().numpy())
        self.val_code_idx.append(code_idx.detach().cpu().numpy())

        self.log("val_loss", loss.item(), on_step=True, on_epoch=True, prog_bar=True)

        # for the first validation step, plot the model
        if batch_idx == 0:
            # get loggers
            comet_logger = None
            for logger in self.trainer.loggers:
                if isinstance(logger, L.pytorch.loggers.CometLogger):
                    comet_logger = logger.experiment

            curr_epoch, curr_step = self.trainer.current_epoch, self.trainer.global_step

            plot_dir = Path(self.trainer.default_root_dir + "/plots/")
            plot_dir.mkdir(exist_ok=True)
            plot_filename = f"{plot_dir}/epoch{curr_epoch}_gstep{curr_step}_original_vs_reco.png"
            # log the plot
            plot_model(
                self.model,
                samples=batch["part_features"],
                masks=batch["part_mask"],
                device=self.device,
                saveas=plot_filename,
            )
            if comet_logger is not None:
                comet_logger.log_image(
                    plot_filename, name=plot_filename.split("/")[-1], step=curr_step
                )

        return loss

    def on_test_epoch_start(self) -> None:
        self.test_x_original = []
        self.test_x_reco = []
        self.test_mask = []
        self.test_labels = []
        self.test_code_idx = []

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        loss, x_original, x_reco, mask, labels, code_idx = self.model_step(batch, return_x=True)

        # save the original and reconstructed data
        self.test_x_original.append(x_original.detach().cpu().numpy())
        self.test_x_reco.append(x_reco.detach().cpu().numpy())
        self.test_mask.append(mask.detach().cpu().numpy())
        self.test_labels.append(labels.detach().cpu().numpy())
        self.test_code_idx.append(code_idx.detach().cpu().numpy())

        self.log("test_loss", loss.item(), on_step=True, on_epoch=True, prog_bar=True)

    def tokenize_ak_array(self, ak_arr, pp_dict, batch_size=256, pad_length=128, hide_pbar=False):
        """Tokenize an awkward array of jets.

        Parameters
        ----------
        ak_arr : ak.Array
            Awkward array of jets, shape (N_jets, <var>, N_features).
        pp_dict : dict
            Dictionary with preprocessing information.
        batch_size : int, optional
            Batch size for the evaluation loop. The default is 256.
        pad_length : int, optional
            Length to which the tokens are padded. The default is 128.
        hide_pbar : bool, optional
            Whether to hide the progress bar. The default is False.

        Returns
        -------
        ak.Array
            Awkward array of tokens, shape (N_jets, <var>).
        """

        # preprocess the ak_arrary
        ak_arr = ak_select_and_preprocess(ak_arr, pp_dict=pp_dict)
        ak_arr_padded, mask = ak_pad(ak_arr, maxlen=pad_length, return_mask=True)
        # convert to numpy
        arr = ak_to_np_stack(ak_arr_padded, names=pp_dict.keys())
        # convert to torch tensor
        x = torch.from_numpy(arr).float()
        mask = torch.from_numpy(mask.to_numpy()).float()

        codes = []
        dataset = TensorDataset(x, mask)
        dataloader = DataLoader(dataset, batch_size=batch_size)

        with torch.no_grad():
            if not hide_pbar:
                pbar = tqdm(dataloader)
            else:
                pbar = dataloader
            for i, (x_batch, mask_batch) in enumerate(pbar):
                # move to device
                x_batch = x_batch.to(self.device)
                mask_batch = mask_batch.to(self.device)
                x_particle_reco, vq_out = self.forward(x_batch, mask_batch)
                code = vq_out["q"]
                codes.append(code)
        codes = torch.cat(codes, dim=0).detach().cpu().numpy()
        mask = mask.detach().cpu().numpy()
        tokens = np_to_ak(codes, names=["token"], mask=mask)["token"]
        return tokens

    def reconstruct_ak_tokens(
        self, tokens_ak, pp_dict, batch_size=256, pad_length=128, hide_pbar=False
    ):
        """Reconstruct tokenized awkward array.

        Parameters
        ----------
        tokens_ak : ak.Array
            Awkward array of tokens, shape (N_jets, <var>).
        pp_dict : dict
            Dictionary with preprocessing information.
        batch_size : int, optional
            Batch size for the evaluation loop. The default is 256.
        pad_length : int, optional
            Length to which the tokens are padded. The default is 128.
        hide_pbar : bool, optional
            Whether to hide the progress bar. The default is False.

        Returns
        -------
        ak.Array
            Awkward array of reconstructed jets, shape (N_jets, <var>, N_features).
        """

        self.model.eval()

        tokens, mask = ak_pad(tokens_ak, maxlen=pad_length, return_mask=True)
        tokens = torch.from_numpy(tokens.to_numpy()).long()
        mask = torch.from_numpy(mask.to_numpy()).float()

        x_reco = []
        dataset = TensorDataset(tokens, mask)
        dataloader = DataLoader(dataset, batch_size=batch_size)

        codebook = self.model.vqlayer.codebook.weight

        # if the codebook has an affine transform, apply it
        # before using it to reconstruct the data
        # see https://github.com/minyoungg/vqtorch/blob/main/vqtorch/nn/vq.py#L102-L104
        if hasattr(self.model.vqlayer, "affine_transform"):
            codebook = self.model.vqlayer.affine_transform(codebook)

        last_batch = None
        with torch.no_grad():
            if not hide_pbar:
                pbar = tqdm(dataloader)
            else:
                pbar = dataloader
            for i, (tokens_batch, mask_batch) in enumerate(pbar):
                # move to device
                tokens_batch = tokens_batch.to(self.device)
                mask_batch = mask_batch.to(self.device)
                try:
                    z_q = F.embedding(tokens_batch, codebook)
                except Exception as e:  # noqa: E722
                    print(f"Error in embedding: {e}")
                    print("batch shape", tokens_batch.shape)
                    print("batch max", tokens_batch.max())
                    print("batch min", tokens_batch.min())

                if last_batch is not None:
                    break

                if hasattr(self.model, "latent_projection_out"):
                    x_reco_batch = self.model.latent_projection_out(z_q) * mask_batch.unsqueeze(-1)
                    x_reco_batch = self.model.decoder_normformer(x_reco_batch, mask=mask_batch)
                    x_reco_batch = self.model.output_projection(
                        x_reco_batch
                    ) * mask_batch.unsqueeze(-1)
                elif hasattr(self.model, "decoder"):
                    x_reco_batch = self.model.decoder(z_q)
                else:
                    raise ValueError("Unknown model structure. Cannot reconstruct.")
                x_reco.append(x_reco_batch)

        x_reco = torch.cat(x_reco, dim=0).detach().cpu().numpy()
        x_reco_ak = np_to_ak(x_reco, names=pp_dict.keys(), mask=mask.detach().cpu().numpy())
        x_reco_ak = ak_select_and_preprocess(x_reco_ak, pp_dict, inverse=True)

        return x_reco_ak

    def on_validation_epoch_end(self) -> None:
        """Lightning hook that is called when a validation epoch ends."""
        self.val_x_original_concat = np.concatenate(self.val_x_original)
        self.val_x_reco_concat = np.concatenate(self.val_x_reco)
        self.val_mask_concat = np.concatenate(self.val_mask)
        self.val_labels_concat = np.concatenate(self.val_labels)
        self.val_code_idx_concat = np.concatenate(self.val_code_idx)

    def on_test_epoch_end(self):
        self.test_x_original_concat = np.concatenate(self.test_x_original)
        self.test_x_reco_concat = np.concatenate(self.test_x_reco)
        self.test_mask_concat = np.concatenate(self.test_mask)
        self.test_labels_concat = np.concatenate(self.test_labels)
        self.test_code_idx_concat = np.concatenate(self.test_code_idx)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Configures optimizers and learning-rate schedulers to be used for training."""
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }

        return {"optimizer": optimizer}


def plot_model(model, samples, device="cuda", n_examples_to_plot=200, masks=None, saveas=None):
    """Visualize the model.

    Parameters
    ----------
    model : nn.Module
        The model.
    samples : Tensor
        The input data.
    device : str, optional
        Device to use. The default is "cuda".
    n_examples_to_plot : int, optional
        Number of examples to plot. The default is 200.
    """

    samples = samples.to(device)
    model = model.to(device)

    # run the model on the input data
    with torch.no_grad():
        # print(f"Model device: {next(model.parameters()).device}")
        # print(f"Samples device: {samples.device}")
        r, vq_out = model(samples, masks)
        z_q = vq_out["z_q"]
        z_e = vq_out["z"]
        idx = vq_out["q"]

        if masks is not None:
            r = r[masks == 1]
            z_e = z_e[masks == 1]
            z_q = z_q[masks == 1]
            idx = idx[masks == 1]

        z_e = z_e.squeeze(1)
        z_q = z_q.squeeze(1)
        idx = idx.squeeze(1)

        # move r, z_e, z_q, idx to cpu for plotting
        r = r.detach().cpu()
        z_e = z_e.detach().cpu()
        z_q = z_q.detach().cpu()
        idx = idx.detach().cpu()

    samples = samples.detach().cpu().numpy()
    if masks is not None:
        masks = masks.detach().cpu().numpy()
        samples = samples[masks == 1]

    # create detached copy of the codebook to plot this
    fig, axarr = plt.subplots(1, 5, figsize=(15, 3))
    # axarr = axarr.flatten()

    style_tokens = dict(color="forestgreen")
    style_true = dict(color="royalblue")
    style_tokens_emb = dict(color="darkorange")
    style_true_emb = dict(color="darkorchid")

    ax = axarr[0]
    ax.scatter(
        z_e[:n_examples_to_plot, 0],
        z_e[:n_examples_to_plot, 1],
        alpha=0.4,
        marker="o",
        label="Samples",
        **style_true_emb,
    )
    ax.scatter(
        z_q[:n_examples_to_plot, 0],
        z_q[:n_examples_to_plot, 1],
        alpha=0.6,
        marker="x",
        label="Closest tokens",
        **style_tokens_emb,
    )
    ax.set_xlabel("$e_1$")
    ax.set_ylabel("$e_2$")
    ax.legend(loc="upper right")
    ax.set_title("Embeddings \n(samples and closest tokens)")

    ax = axarr[1]
    ax.scatter(
        z_e[:n_examples_to_plot, 0],
        z_e[:n_examples_to_plot, 2],
        alpha=0.2,
        s=26,
        **style_true_emb,
        label="Samples",
    )
    ax.scatter(
        z_q[:n_examples_to_plot, 0],
        z_q[:n_examples_to_plot, 2],
        alpha=0.7,
        s=26,
        **style_tokens_emb,
        marker="x",
        label="Closest tokens",
    )
    ax.set_xlabel("$e_1$")
    ax.set_ylabel("$e_3$")
    ax.set_title("Embeddings \n(samples and closest token)")
    ax.legend(loc="upper right")

    # plot the original sample and the reconstructed sample (the first sample in the batch)
    # plot original sample
    ax = axarr[2]
    ax.scatter(
        samples[:n_examples_to_plot, 0],
        samples[:n_examples_to_plot, 1],
        alpha=0.2,
        s=26,
        **style_true,
        label="Original",
    )
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_title("Original constituents \n(first few in batch)")
    # plot reconstructed sample
    ax.scatter(
        r[:n_examples_to_plot, 0],
        r[:n_examples_to_plot, 1],
        alpha=0.7,
        s=26,
        marker="x",
        **style_tokens,
        label="Reco. token",
    )
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_title("Data space \nTrue vs reconstructed")
    ax.legend(loc="upper right")

    # plot true vs reconstructed for deltaR and ptrel
    ax = axarr[3]
    ax.scatter(
        samples[:n_examples_to_plot, 0],
        samples[:n_examples_to_plot, 2],
        s=26,
        alpha=0.2,
        **style_true,
        label="Original",
    )
    ax.scatter(
        r[:n_examples_to_plot, 0],
        r[:n_examples_to_plot, 2],
        s=26,
        alpha=0.7,
        **style_tokens,
        marker="x",
        label="Reco. tokens",
    )
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_3$")
    ax.legend(loc="upper right")
    ax.set_title("Data space \nTrue vs reconstructed")

    # plot the histogram of the codebook indices (i.e. a codebook_size x codebook_size
    # histogram with each entry in the histogram corresponding to one sample associated
    # with the corresponding codebook entry)
    ax = axarr[4]
    n_codes = model.vq_kwargs["num_codes"]
    bins = np.linspace(-0.5, n_codes + 0.5, n_codes + 1)
    ax.hist(idx, bins=bins)
    ax.set_title(
        "Codebook histogram\n(Each entry corresponds to one sample\nbeing associated with that"
        " codebook entry)",
        fontsize=8,
    )

    # make empty axes invisible
    def is_axes_empty(ax):
        return not (
            ax.lines
            or ax.patches
            or ax.collections
            or ax.images
            or ax.texts
            or ax.artists
            or ax.tables
        )

    for ax in axarr.flatten():
        if is_axes_empty(ax):
            ax.set_visible(False)

    fig.tight_layout()
    plt.show()
    if saveas is not None:
        fig.savefig(saveas)


def plot_loss(loss_history, lr_history, moving_average=100):
    if len(loss_history) < moving_average:
        print("Not enough steps to plot loss history")
        return
    fig, ax1 = plt.subplots(figsize=(5, 2))
    ax2 = ax1.twinx()

    # Plot loss history
    loss_history = np.array(loss_history)
    loss_history = np.convolve(loss_history, np.ones(moving_average), "valid") / moving_average
    ax1.plot(loss_history, color="blue")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Loss")
    ax1.set_yscale("log")
    ax1.grid(True, which="both", ls="-", alpha=0.5)
    ax1.set_title(f"Loss history (moving average over {moving_average} steps)", fontsize=8)

    # Plot lr history
    ax2.plot(lr_history, color="red")
    ax2.set_ylabel("Learning Rate")

    fig.tight_layout()
    plt.show()
