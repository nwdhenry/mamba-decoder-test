"""Minimal NanoGPT implementation used for testing.

This module provides a very small GPT model that mirrors the interface of the
original `nanoGPT` implementation but is lightweight enough to be used inside
the test suite. It avoids any heavy dependencies and is designed to run purely
on CPU.
"""

from dataclasses import dataclass
import torch
from torch import nn

@dataclass
class GPTConfig:
    """Configuration for :class:`GPT`.

    Parameters
    ----------
    vocab_size : int, default=65
        Size of the vocabulary.
    block_size : int, default=8
        Maximum context length supported by the model.
    n_layer : int, default=1
        Number of transformer encoder layers.
    n_head : int, default=2
        Number of attention heads.
    n_embd : int, default=16
        Embedding dimension.
    """

    vocab_size: int = 65
    block_size: int = 8
    n_layer: int = 1
    n_head: int = 2
    n_embd: int = 16

class GPT(nn.Module):
    """Tiny GPT model used in tests."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.token_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)
        layer = nn.TransformerEncoderLayer(
            config.n_embd, config.n_head, 4 * config.n_embd
        )
        self.transformer = nn.TransformerEncoder(layer, config.n_layer)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """Compute logits for the given token indices.

        Parameters
        ----------
        idx : torch.Tensor
            Input token ids of shape ``(batch, seq_len)``.

        Returns
        -------
        torch.Tensor
            Logits of shape ``(batch, seq_len, vocab_size)``.
        """

        b, t = idx.shape
        assert t <= self.config.block_size
        pos = torch.arange(t, device=idx.device)
        x = self.token_emb(idx) + self.pos_emb(pos)
        x = self.transformer(x)
        logits = self.lm_head(x)
        return logits
