import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import vector

vector.register_awkward()

logger = logging.getLogger(__name__)


class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, n_heads, attention_dropout):
        super().__init__()
        assert embedding_dim % n_heads == 0, "Embedding dim must be divisible by number of heads"

        self.head_dim = embedding_dim // n_heads
        self.n_heads = n_heads
        self.embedding_dim = embedding_dim

        self.key = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.query = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.value = nn.Linear(embedding_dim, embedding_dim, bias=False)

        # Create a causal attention mask and store it as self.tril. Being a
        # buffer means that it will not be included as parameters in the model.
        self.register_buffer("tril", torch.tril(torch.ones(embedding_dim, embedding_dim)))
        self.dropout = nn.Dropout(attention_dropout)

        self.proj = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, x, padding_mask=None):
        B, T, C = x.shape
        # input of size (batch, time-step, channels); channels = embedding dimension
        # output of size (batch, time-step, embedding_dim)

        k = self.key(x)  # (B, T, E)
        q = self.query(x)  # (B, T, E)
        v = self.value(x)  # (B, T, E)

        # We implicitly split the matrix by adding a `num_heads` dimension
        # Unroll last dim: (B, T, E) -> (B, T, num_heads, head_dim)
        k = k.view(B, T, self.n_heads, self.head_dim)
        v = v.view(B, T, self.n_heads, self.head_dim)
        q = q.view(B, T, self.n_heads, self.head_dim)

        # Transpose: (B, T, n_heads, head_dim) -> (B, n_heads, T, head_dim)
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        # Compute scaled dot-product attention
        # (B, n_heads, T, head_dim) @ (B, n_heads, head_dim, T) -> (B, n_heads, T, T)
        attn_scores = q @ k.transpose(2, 3) * k.shape[-1] ** -0.5

        if padding_mask is not None:
            padding_mask = padding_mask.unsqueeze(-1).expand(-1, -1, T)  # (B, T) -> (B, T, T)
            # (B, T, T) -> (B, n_heads, T, T)
            padding_mask = padding_mask.unsqueeze(1).expand(B, self.n_heads, T, T)
            # Need to set a finite number for the masking, instead of -inf,
            # otherwise softmax results in nans.
            # (B, n_heads, T, T)
            attn_scores = attn_scores.masked_fill(padding_mask == 0, float("-1e9"))

        # Apply the causal mask, cropped to the sequence length
        # (B, n_heads, T, T)
        attn_scores = attn_scores.masked_fill(self.tril[:T, :T] == 0, float("-inf"))

        attn_weights = F.softmax(attn_scores, dim=-1)  # (B, n_heads, T, T)
        attn_weights = self.dropout(attn_weights)

        # attn_weights have shape (B, n_heads, T, T) and v (B, n_heads, T, head_dim)
        # (B, n_heads, T, head_dim) -> (B, T, n_heads, head_dim)
        context_vec = (attn_weights @ v).transpose(1, 2)

        # Combine heads, where embedding_dim = n_heads * head_dim
        context_vec = context_vec.contiguous().view(B, T, self.embedding_dim)
        context_vec = self.proj(context_vec)

        return context_vec


class FeedForward(nn.Module):
    """Simple linear layer followed by a non-linearity to be placed after the attention blocks."""

    def __init__(self, embedding_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, 4 * embedding_dim),
            nn.ReLU(),
            nn.Linear(4 * embedding_dim, embedding_dim),
            # nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class GPT_DecoderBlock(nn.Module):
    """The GPT decoder block."""

    def __init__(self, embedding_dim, attention_dropout, n_heads, verbose=False):
        super().__init__()
        self.verbose = verbose
        self.mha_block = MultiHeadAttention(embedding_dim, n_heads, attention_dropout)
        self.ff_block = FeedForward(embedding_dim)
        self.layernorm_1 = nn.LayerNorm(embedding_dim)
        self.layernorm_2 = nn.LayerNorm(embedding_dim)

    def forward(self, x, padding_mask=None):
        x_residual = x

        x = self.mha_block(x, padding_mask=padding_mask)
        x += x_residual

        x = self.layernorm_1(x)
        x_residual = x

        x = self.ff_block(x)
        x += x_residual

        x = self.layernorm_2(x)

        return x


class BackboneModel(nn.Module):
    """Model that is used as the backbone in our studies.

    Going from integer tokens to embeddings via an embedding table, then through a stack of GPT
    blocks. The output is the final embeddings.
    """

    def __init__(
        self,
        embedding_dim,
        attention_dropout,
        vocab_size,
        max_sequence_len,
        n_heads,
        n_GPT_blocks,
        n_classes=2,
        classify=False,
        verbosity=True,
        n_tokens=None,
        return_embeddings=False,  # only there for now for backwards-compatibility with the old model
        **kwargs,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.max_sequence_len = max_sequence_len
        self.verbose = verbosity
        self.embedding_table = nn.Embedding(vocab_size, embedding_dim)
        self.return_embeddings = return_embeddings

        GPT_block_stack = []
        for _ in range(n_GPT_blocks):
            GPT_block_stack.extend(
                [
                    GPT_DecoderBlock(
                        embedding_dim,
                        attention_dropout,
                        n_heads=n_heads,
                        verbose=self.verbose,
                    )
                ]
            )
        self.GPT_blocks = nn.Sequential(*GPT_block_stack)

    def forward(self, x, padding_mask=None):
        x = self.embedding_table(x)

        for block in self.GPT_blocks:
            x = block(x, padding_mask=padding_mask)

        return x
