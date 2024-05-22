"""Backbone model with different heads."""

import logging
import time
from typing import Any, Dict, Tuple

import awkward as ak
import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import vector
from tqdm import tqdm

from gabbro.metrics.utils import calc_accuracy
from gabbro.models.gpt_model import BackboneModel

vector.register_awkward()

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------
# ------------ BACKBONE + Generative (next-token-prediction) head ---------
# -------------------------------------------------------------------------


class NextTokenPredictionHead(nn.Module):
    """Head for predicting the next token in a sequence."""

    def __init__(self, embedding_dim, vocab_size):
        super().__init__()
        self.fc1 = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x):
        return self.fc1(x)


class BackboneNextTokenPredictionLightning(L.LightningModule):
    """PyTorch Lightning module for training the backbone model."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler = None,
        model_kwargs={},
        token_dir=None,
        verbose=False,
        **kwargs,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        # initialize the backbone
        self.module = BackboneModel(**model_kwargs)

        # initialize the model head
        self.head = NextTokenPredictionHead(
            embedding_dim=model_kwargs["embedding_dim"],
            vocab_size=model_kwargs["vocab_size"],
        )

        # initialize the loss function
        self.criterion = nn.CrossEntropyLoss()

        self.token_dir = token_dir
        self.verbose = verbose

        self.train_loss_history = []
        self.val_loss_list = []

        self.validation_cnt = 0
        self.validation_output = {}

        self.backbone_weights_path = model_kwargs.get("backbone_weights_path", "None")

        print(f"Backbone weights path: {self.backbone_weights_path}")

        if self.backbone_weights_path is not None:
            if self.backbone_weights_path != "None":
                self.load_backbone_weights(self.backbone_weights_path)

    def load_backbone_weights(self, ckpt_path):
        print(f"Loading backbone weights from {ckpt_path}")
        ckpt = torch.load(ckpt_path)
        state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
        self.load_state_dict(state_dict, strict=False)

    def forward(self, x, mask=None):
        if self.module.return_embeddings:
            backbone_out = self.module(x, mask)
            logits = self.head(backbone_out)
        else:
            logits = self.module(x, mask)
        if self.verbose:
            print("Logits shape: ", logits.shape)
        return logits

    def model_step(self, batch, return_logits=False):
        """Perform a single model step on a batch of data.

        Parameters
        ----------
        batch : dict
            A batch of data as a dictionary containing the input and target tensors,
            as well as the mask.
        return_logits : bool, optional
            Whether to return the logits or not. (default is False)
        """

        # all token-ids up to the last one are the input, the ones from the second
        # to the (including) last one are the target
        # this model step uses the convention that the first particle feature
        # is the token, with the tokens up to the last one
        # the second particle feature is the target token (i.e. the next token)

        X = batch["part_features"]
        X = X.squeeze().long()
        input = X[:, :, 0]
        targets = X[:, :, 1]
        mask = batch["part_mask"]

        # compute the logits (i.e. the predictions for the next token)
        logits = self.forward(input, mask)

        # reshape the logits and targets to work with the loss function
        B, T, C = logits.shape
        logits = logits.view(B * T, C)
        targets = targets.contiguous().view(B * T)

        loss = self.criterion(logits, targets)

        if return_logits:
            return loss, X, logits, mask, targets

        return loss

    @torch.no_grad()
    def generate_batch(self, batch_size):
        """Generate a batch of jet constituents autoregressively.

        Parameters
        ----------
        batch_size : int
            Number of jets to generate.

        Returns
        -------
        ak.Array
            The generated jets (i.e. their token ids, in the shape (batch_size, <var>).
        """
        # idx is (B, T) array of indices in the current context, initialized with the start token
        # thus idx has shape (B, 1) at the beginning
        device = next(self.module.parameters()).device  # get the device of the model
        idx = torch.zeros(batch_size, 1).long().to(device)

        for i in range(self.module.max_sequence_len):
            # get the predictions for the next token
            logits = self(idx)
            print("Logit shape input for generation: ", logits.shape) if self.verbose else None
            # only look at next-token prediction of last token
            logits = logits[:, -1, :]  # (B, T, C) becomes (B, C)
            # apply softmax to get probabilities, and exclude the start-token (index 0)
            # (otherwise it can happen, that the start token is prediced as the next token)
            probs = F.softmax(logits[:, 1:], dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) + 1  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
            print(
                "appended idx_next to original idx, shape: ", idx.shape
            ) if self.verbose else None

        # TODO: the thing below gets the job done, but could be improved
        # remove everything after the first stop token if it exists in the jet
        gen_batch_np = idx.detach().cpu().numpy()
        gen_batch_ak = ak.from_numpy(gen_batch_np)
        gen_batch_until_stop = []

        # loop over the jets in the batch, and only keep the tokens until the stop token
        for jet in gen_batch_ak:
            stop_token_position = np.where(jet == self.module.vocab_size - 1)
            if len(stop_token_position[0]) > 0:
                stop_token_position = stop_token_position[0][0]
            else:
                stop_token_position = jet.shape[0]
            gen_batch_until_stop.append(jet[:stop_token_position])

        return ak.Array(gen_batch_until_stop)

    def generate_n_jets_batched(self, n_jets, batch_size, saveas=None):
        """Generate jets in batches.

        Parameters
        ----------
        n_jets : int
            Number of jets to generate.
        batch_size : int
            Batch size to use during generation (use as large as possible with memory.)
        saveas : str, optional
            Path to save the generated jets to (in parquet format). (default is None)

        Returns
        -------
        ak.Array
            The generated jets (i.e. their token ids, in the shape (n_jets, <var>).
        """
        n_batches = n_jets // batch_size + 1
        generated_jets = []

        print(f"Generating {n_jets} jets in {n_batches} batches of size {batch_size}")

        for i in tqdm(range(n_batches)):
            gen_batch_ak = self.generate_batch(batch_size)
            generated_jets.append(gen_batch_ak)

        # concatenate the generated batches
        generated_jets = ak.concatenate(generated_jets)[:n_jets]

        if saveas is not None:
            print(f"Saving generated jets to {saveas}")
            ak.to_parquet(generated_jets, saveas)

        return generated_jets

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
        self.val_token_ids_list = []
        self.val_token_masks_list = []

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        loss, X, logits, mask, targets = self.model_step(batch, return_logits=True)

        self.val_token_ids_list.append(batch["part_features"].float().detach().cpu().numpy())
        self.val_token_masks_list.append(batch["part_mask"].float().detach().cpu().numpy())
        self.log("val_loss", loss.item(), on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def on_test_epoch_start(self) -> None:
        pass

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        loss, X, logits, mask, targets = self.model_step(batch, return_logits=True)
        self.log("test_loss", loss.item(), on_step=True, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        """Lightning hook that is called when a validation epoch ends."""
        pass

    def on_test_epoch_end(self):
        pass

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


# -------------------------------------------------------------------------
# ------------------ BACKBONE + Classification head -----------------------
# -------------------------------------------------------------------------


class NormformerCrossBlock(nn.Module):
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

    def forward(self, x, class_token, mask=None, return_attn_weights=False):
        # x: (B, S, F)
        # mask: (B, S)
        x = x * mask.unsqueeze(-1)

        # calculate cross-attention
        x_norm = self.norm1(x)
        attn_output, attn_weights = self.attn(
            query=class_token, key=x_norm, value=x_norm, key_padding_mask=mask != 1
        )
        return attn_output


class ClassifierNormformer(torch.nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        num_heads=4,
        dropout_rate=0.1,
        num_class_blocks=3,
        model_kwargs={"n_out_nodes": 2, "fc_params": [(100, 0.1), (100, 0.1)]},
        **kwargs,
    ):
        super().__init__()

        self.model_kwargs = model_kwargs
        self.n_out_nodes = model_kwargs["n_out_nodes"]
        self.dropout_rate = dropout_rate
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_class_blocks = num_class_blocks
        self.class_token = nn.Parameter(torch.randn(1, 1, hidden_dim))

        self.input_projection = nn.Linear(self.input_dim, self.hidden_dim)
        self.class_attention_blocks = nn.ModuleList(
            [
                NormformerCrossBlock(
                    input_dim=self.hidden_dim,
                    num_heads=self.num_heads,
                    dropout_rate=self.dropout_rate,
                    mlp_dim=self.hidden_dim,
                )
                for _ in range(self.num_class_blocks)
            ]
        )
        self.final_mlp = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_dim, self.model_kwargs["n_out_nodes"]),
        )

        self.loss_history = []
        self.lr_history = []

    def forward(self, x, mask):
        # expand class token and add to mask
        class_token = self.class_token.expand(x.size(0), -1, -1)
        mask_with_token = torch.cat([torch.ones(x.size(0), 1).to(x.device), mask], dim=1)

        # pass through class attention blocks, always use the updated class token
        for block in self.class_attention_blocks:
            x_class_token_and_x_encoded = torch.cat([class_token, x], dim=1)
            # class_token = block(x_class_token_and_x_encoded, mask=mask_with_token)[:, :1, :]
            class_token = block(x_class_token_and_x_encoded, class_token, mask=mask_with_token)

        # pass through final mlp
        class_token = self.final_mlp(class_token).squeeze(1)
        return class_token


class ClassificationHead(torch.nn.Module):
    """Classification head for the backbone model."""

    def __init__(self, model_kwargs={"n_out_nodes": 2}):
        super().__init__()
        self.backbone_weights_path = None

        if "n_out_nodes" not in model_kwargs:
            model_kwargs["n_out_nodes"] = 2
        if "return_embeddings" not in model_kwargs:
            model_kwargs["return_embeddings"] = True

        self.n_out_nodes = model_kwargs["n_out_nodes"]
        model_kwargs.pop("n_out_nodes")

        self.classification_head_linear_embed = nn.Linear(
            model_kwargs["embedding_dim"],
            model_kwargs["embedding_dim"],
        )
        self.classification_head_linear_class = nn.Linear(
            model_kwargs["embedding_dim"],
            self.n_out_nodes,
        )

    def forward(self, x, mask):
        embeddings = F.relu(self.classification_head_linear_embed(x))
        embeddings_sum = torch.sum(embeddings * mask.unsqueeze(-1), dim=1)
        logits = self.classification_head_linear_class(embeddings_sum)
        return logits


class BackboneClassificationLightning(L.LightningModule):
    """Backbone with classification head."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler = None,
        class_head_type: str = "summation",
        model_kwargs: dict = {},
        **kwargs,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        # initialize the backbone
        self.module = BackboneModel(**model_kwargs)

        # initialize the model head
        if class_head_type == "summation":
            self.head = ClassificationHead(
                model_kwargs={
                    "n_out_nodes": model_kwargs["n_out_nodes"],
                    "embedding_dim": model_kwargs["embedding_dim"],
                }
            )
        elif class_head_type == "class_attention":
            self.head = ClassifierNormformer(
                input_dim=model_kwargs["embedding_dim"],
                hidden_dim=model_kwargs["embedding_dim"],
                model_kwargs={"n_out_nodes": model_kwargs["n_out_nodes"]},
                num_heads=2,
                num_class_blocks=3,
                dropout_rate=0.1,
            )
        else:
            raise ValueError(f"Invalid class_head_type: {class_head_type}")

        self.criterion = torch.nn.CrossEntropyLoss()

        self.train_loss_history = []
        self.val_loss_history = []

        self.backbone_weights_path = model_kwargs.get("backbone_weights_path", "None")
        print(f"Backbone weights path: {self.backbone_weights_path}")

        if self.backbone_weights_path is not None:
            if self.backbone_weights_path != "None":
                self.load_backbone_weights(self.backbone_weights_path)

    def load_backbone_weights(self, ckpt_path):
        print(f"Loading backbone weights from {ckpt_path}")
        ckpt = torch.load(ckpt_path)
        state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
        self.load_state_dict(state_dict, strict=False)

    def forward(self, X, mask):
        embeddings = self.module(X, mask)
        logits = self.head(embeddings, mask)
        return logits

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        pass

    def on_train_epoch_start(self) -> None:
        self.train_preds_list = []
        self.train_labels_list = []
        print(f"Epoch {self.trainer.current_epoch} started.", end="\r")

    def model_step(self, batch) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        X = batch["part_features"]
        mask = batch["part_mask"]
        jet_labels = batch["jet_type_labels"]
        if len(X.size()) == 2:
            X = X.unsqueeze(-1)
        X = X.squeeze().long()
        # one-hot encode the labels
        logits = self.forward(X, mask)
        labels = F.one_hot(jet_labels.squeeze(), num_classes=self.head.n_out_nodes).float()
        loss = self.criterion(logits.to("cuda"), labels.to("cuda"))
        return loss, logits, labels

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        loss, logits, targets = self.model_step(batch)

        preds = torch.softmax(logits, dim=1)
        self.train_preds_list.append(preds.float().detach().cpu().numpy())
        self.train_labels_list.append(targets.float().detach().cpu().numpy())
        self.train_loss_history.append(loss.float().detach().cpu().numpy())

        acc = calc_accuracy(
            preds=preds.float().detach().cpu().numpy(),
            labels=targets.float().detach().cpu().numpy(),
        )

        self.log(
            "train_loss", loss.item(), on_step=True, on_epoch=True, prog_bar=True, sync_dist=True
        )
        self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

        return loss

    def on_train_epoch_end(self):
        self.train_preds = np.concatenate(self.train_preds_list)
        self.train_labels = np.concatenate(self.train_labels_list)
        print(f"Epoch {self.trainer.current_epoch} finished.", end="\r")
        plt.plot(self.train_loss_history)

    def on_validation_epoch_start(self) -> None:
        self.val_preds_list = []
        self.val_labels_list = []

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        loss, logits, targets = self.model_step(batch)
        preds = torch.softmax(logits, dim=1)
        self.val_preds_list.append(preds.float().detach().cpu().numpy())
        self.val_labels_list.append(targets.float().detach().cpu().numpy())
        # update and log metrics
        acc = calc_accuracy(
            preds=preds.float().detach().cpu().numpy(),
            labels=targets.float().detach().cpu().numpy(),
        )
        self.log(
            "val_loss", loss.item(), on_step=True, on_epoch=True, prog_bar=True, sync_dist=True
        )
        self.log("val_acc", acc, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

    def on_validation_epoch_end(self) -> None:
        """Lightning hook that is called when a validation epoch ends."""
        self.val_preds = np.concatenate(self.val_preds_list)
        self.val_labels = np.concatenate(self.val_labels_list)

    def on_test_start(self):
        self.test_loop_preds_list = []
        self.test_loop_labels_list = []

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set."""
        loss, logits, targets = self.model_step(batch)
        preds = torch.softmax(logits, dim=1)
        self.test_loop_preds_list.append(preds.float().detach().cpu().numpy())
        self.test_loop_labels_list.append(targets.float().detach().cpu().numpy())

        acc = calc_accuracy(
            preds=preds.float().detach().cpu().numpy(),
            labels=targets.float().detach().cpu().numpy(),
        )
        self.log(
            "test_loss", loss.item(), on_step=True, on_epoch=True, prog_bar=True, sync_dist=True
        )
        self.log("test_acc", acc, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

    def on_test_epoch_end(self):
        self.test_preds = np.concatenate(self.test_loop_preds_list)
        self.test_labels = np.concatenate(self.test_loop_labels_list)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Configures optimizers and learning-rate schedulers to be used for training."""
        if self.hparams.model_kwargs.keep_backbone_fixed:
            print("--- Keeping backbone fixed. ---")
            optimizer = self.hparams.optimizer(
                [
                    {"params": self.module.parameters(), "lr": 0.0},
                    {"params": self.head.parameters()},
                ]
            )
        else:
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
