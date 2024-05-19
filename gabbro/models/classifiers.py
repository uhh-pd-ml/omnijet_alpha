from typing import Any, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule
from weaver.nn.model.ParticleTransformer import ParticleTransformer  # noqa: E402

from gabbro.metrics.utils import calc_accuracy
from gabbro.models.vqvae import NormformerStack


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


# --------------------------- Particle Flow Network ---------------------------
class ParticleFlow(nn.Module):
    """Definition of the Particle Flow Network."""

    def __init__(
        self,
        input_dim=None,
        n_out_nodes=2,
        n_embed=16,
        n_tokens=None,
        **kwargs,
    ):
        """Initialise Particle Flow Network.

        Parameters
        ----------
        input_dim : int, optional
            Number of features per point.
        n_out_nodes : int, optional
            Number of output nodes.
        n_embed : int, optional
            Number of embedding dimensions, only used if n_tokens is not None.
        n_tokens : int, optional
            Number of codebook entries (i.e. number of different tokens), only
            used if input_dim is None.
        """

        super().__init__()

        if input_dim is None and n_tokens is None:
            raise ValueError("Either input_dim or n_tokens must be specified")

        self.n_out_nodes = n_out_nodes
        self.n_tokens = n_tokens
        self.n_embed = n_embed

        if n_tokens is None:
            self.phi_1 = nn.Linear(input_dim, 100)
        else:
            self.embedding = nn.Embedding(n_tokens, n_embed)
            self.phi_1 = nn.Linear(n_embed, 100)

        self.phi_2 = nn.Linear(100, 100)
        self.phi_3 = nn.Linear(100, 256)
        self.F_1 = nn.Linear(256, 100)
        self.F_2 = nn.Linear(100, 100)
        self.F_3 = nn.Linear(100, 100)
        self.output_layer = nn.Linear(100, self.n_out_nodes)

    def forward(self, x, mask):
        batch_size, n_points, n_features = x.size()

        # propagate through phi
        if self.n_tokens is not None:
            x = self.embedding(x).squeeze()
        x = F.relu(self.phi_1(x))
        x = F.relu(self.phi_2(x))
        x = F.relu(self.phi_3(x))

        # sum over points dim.
        x_sum = torch.sum(x * mask[..., None], dim=1)

        # propagate through F
        x = F.relu(self.F_1(x_sum))
        x = F.relu(self.F_2(x))
        x = F.relu(self.F_3(x))

        x_out = self.output_layer(x)

        return x_out


class ClassifierPL(LightningModule):
    """Pytorch-lightning module for jet classification."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler = None,
        model_class_name: str = "ParticleFlow",
        model_kwargs: dict = {},
        **kwargs,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.model_class_name = model_class_name
        if "keep_backbone_fixed" in model_kwargs:
            self.keep_backbone_fixed = model_kwargs["keep_backbone_fixed"]
            model_kwargs.pop("keep_backbone_fixed")
        else:
            self.keep_backbone_fixed = False

        if self.model_class_name == "ParticleFlow":
            self.model = ParticleFlow(**model_kwargs)
        elif self.model_class_name == "ClassifierNormformer":
            self.model = ClassifierNormformer(**model_kwargs)
        elif self.model_class_name == "ParT":
            self.model = ParticleTransformer(**model_kwargs)
        else:
            raise ValueError(f"Model class {model_class_name} not supported.")

        self.criterion = torch.nn.CrossEntropyLoss()

        self.train_loss_history = []
        self.val_loss_history = []

    def forward(self, features, mask):
        return self.model(features, mask)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        pass

    def on_train_epoch_start(self) -> None:
        self.train_preds_list = []
        self.train_labels_list = []
        print(f"Epoch {self.trainer.current_epoch} started.", end="\r")

    def model_step(self, batch) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        Parameters
        ----------
        batch : dict
            A batch of data containing the input tensor of images and target labels.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predicted logits.
            - A tensor of target labels.
        """
        X = batch["part_features"]
        mask = batch["part_mask"]
        jet_labels = batch["jet_type_labels"]
        if len(X.size()) == 2:
            X = X.unsqueeze(-1)
        if self.model_class_name == "BackboneWithClasshead":
            X = X.squeeze().long()
        # one-hot encode the labels
        labels = F.one_hot(jet_labels.squeeze(), num_classes=self.model.n_out_nodes).float()
        logits = self.forward(X, mask)
        loss = self.criterion(logits.to("cuda"), labels.to("cuda"))
        return loss, logits, labels

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set."""
        loss, logits, targets = self.model_step(batch)

        preds = torch.softmax(logits, dim=1)
        self.train_preds_list.append(preds.detach().cpu().numpy())
        self.train_labels_list.append(targets.detach().cpu().numpy())
        self.train_loss_history.append(loss.detach().cpu().numpy())

        acc = calc_accuracy(
            preds=preds.detach().cpu().numpy(), labels=targets.detach().cpu().numpy()
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
        self.val_preds_list.append(preds.detach().cpu().numpy())
        self.val_labels_list.append(targets.detach().cpu().numpy())
        # update and log metrics
        acc = calc_accuracy(
            preds=preds.detach().cpu().numpy(), labels=targets.detach().cpu().numpy()
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
        self.test_loop_preds_list.append(preds.detach().cpu().numpy())
        self.test_loop_labels_list.append(targets.detach().cpu().numpy())

        acc = calc_accuracy(
            preds=preds.detach().cpu().numpy(), labels=targets.detach().cpu().numpy()
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
        if self.keep_backbone_fixed:
            print("--- Keeping backbone fixed. ---")
            optimizer = self.hparams.optimizer(
                [
                    {"params": self.model.module.parameters(), "lr": 0.0},
                    {"params": self.model.classification_head_linear_embed.parameters()},
                    {"params": self.model.classification_head_linear_class.parameters()},
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


class ClassifierNormformer(torch.nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        num_heads=1,
        num_enc_blocks=2,
        class_head_kwargs={"n_out_nodes": 2, "fc_params": [(100, 0.1), (100, 0.1)]},
        dropout_rate=0.1,
        num_class_blocks=3,
        **kwargs,
    ):
        super().__init__()

        self.dropout_rate = dropout_rate
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_enc_blocks = num_enc_blocks
        self.num_class_blocks = num_class_blocks
        self.class_head_kwargs = class_head_kwargs
        self.class_token = nn.Parameter(torch.randn(1, 1, hidden_dim))

        self.input_projection = nn.Linear(self.input_dim, self.hidden_dim)
        self.encoder_normformer = NormformerStack(
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            num_blocks=self.num_enc_blocks,
            dropout_rate=self.dropout_rate,
        )
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
        self.initialize_classification_head()

        self.loss_history = []
        self.lr_history = []

    def forward(self, x, mask):
        # encode
        x = self.input_projection(x)
        x_encoded = self.encoder_normformer(x, mask=mask)
        # concatenate class token and x
        class_token = self.class_token.expand(x.size(0), -1, -1)
        mask_with_token = torch.cat([torch.ones(x.size(0), 1).to(x.device), mask], dim=1)

        # pass through class attention blocks, always use the updated class token
        for block in self.class_attention_blocks:
            x_class_token_and_x_encoded = torch.cat([class_token, x_encoded], dim=1)
            class_token = block(x_class_token_and_x_encoded, class_token, mask=mask_with_token)

        return self.classification_head(class_token.squeeze(1))

    def initialize_classification_head(self):
        if self.class_head_kwargs is None:
            self.class_head_kwargs = {
                "fc_params": [
                    [128, 0.1],
                    [128, 0.1],
                ],
                "n_out_nodes": 2,
            }

        fc_params = [[self.hidden_dim, 0]] + self.class_head_kwargs["fc_params"]
        self.n_out_nodes = self.class_head_kwargs["n_out_nodes"]

        layers = []

        for i in range(1, len(fc_params)):
            in_dim = fc_params[i - 1][0]
            out_dim = fc_params[i][0]
            dropout_rate = fc_params[i][1]
            layers.extend(
                [
                    nn.Linear(in_dim, out_dim),
                    nn.Dropout(dropout_rate),
                    nn.ReLU(),
                ]
            )
        # add final layer
        layers.extend([nn.Linear(fc_params[-1][0], self.n_out_nodes)])

        self.classification_head = nn.Sequential(*layers)
