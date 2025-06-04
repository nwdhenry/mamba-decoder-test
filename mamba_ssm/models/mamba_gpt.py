# Copyright (c) 2024, Tri Dao, Albert Gu.
"""Simple decoder-only model using Mamba blocks."""

from dataclasses import dataclass
from typing import Optional, Iterable

import torch
from torch import nn

from mamba_ssm.modules.mamba_simple import Mamba


@dataclass
class MambaGPTConfig:
    """Configuration for :class:`MambaGPT`."""

    vocab_size: int = 50257
    d_model: int = 768
    n_layer: int = 12
    # kwargs to pass to each Mamba block
    mamba_kwargs: Optional[dict] = None
    tie_embeddings: bool = True



class MambaBlock(nn.Module):
    """LayerNorm -> Mamba with residual connection."""

    def __init__(self, d_model: int, layer_idx: int, mamba_kwargs: Optional[dict], *, device=None, dtype=None) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.norm = nn.LayerNorm(d_model, **factory_kwargs)
        self.mamba = Mamba(d_model, layer_idx=layer_idx, **(mamba_kwargs or {}), **factory_kwargs)

    def forward(self, hidden_states: torch.Tensor, *, inference_params=None) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.norm(hidden_states)
        hidden_states = self.mamba(hidden_states, inference_params=inference_params)
        hidden_states = hidden_states + residual
        return hidden_states

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mamba.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)


class MambaGPT(nn.Module):
    """Decoder-only language model built from Mamba blocks."""

    def __init__(self, config: MambaGPTConfig, *, device=None, dtype=None) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.config = config

        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model, **factory_kwargs)
        self.blocks = nn.ModuleList(
            [MambaBlock(config.d_model, i, config.mamba_kwargs, device=device, dtype=dtype) for i in range(config.n_layer)]
        )
        self.norm_f = nn.LayerNorm(config.d_model, **factory_kwargs)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False, **factory_kwargs)

        if config.tie_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

    def forward(self, input_ids: torch.Tensor, *, inference_params=None) -> torch.Tensor:
        """Compute logits for ``input_ids``.

        Args:
            input_ids: ``LongTensor`` of shape ``(batch, seq_len)``
            inference_params: optional parameters for cached inference
        Returns:
            ``FloatTensor`` of shape ``(batch, seq_len, vocab_size)``
        """
        hidden_states = self.embed_tokens(input_ids)
        for block in self.blocks:
            hidden_states = block(hidden_states, inference_params=inference_params)
        hidden_states = self.norm_f(hidden_states)
        logits = self.lm_head(hidden_states)
        return logits

    def num_parameters(self, only_trainable: bool = True) -> int:
        """Return the number of parameters."""
        params: Iterable[torch.Tensor] = (
            p for p in self.parameters() if p.requires_grad or not only_trainable
        )
        return sum(p.numel() for p in params)
