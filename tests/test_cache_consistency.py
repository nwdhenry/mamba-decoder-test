"""Tests ensuring caching yields identical results to full generation."""

import pytest
import torch
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.utils.generation import InferenceParams


def test_cache_consistency():
    """Check that autoregressive generation with cache matches full pass."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = MambaConfig(d_model=64, n_layer=2, vocab_size=100)
    model = MambaLMHeadModel(cfg, device=device)
    prompt = torch.randint(0, cfg.vocab_size, (1, 20), device=device)

    # Run generation without cache
    logits_full = model(prompt).logits

    # Run autoregressive with cache
    inference_params = InferenceParams(max_seqlen=20, max_batch_size=1)
    logits_cached = []
    for i in range(prompt.shape[1]):
        inp = prompt[:, i:i+1]
        logits = model(inp, inference_params=inference_params, num_last_tokens=1).logits
        logits_cached.append(logits)
    logits_cached = torch.cat(logits_cached, dim=1)
    assert torch.allclose(logits_full, logits_cached, atol=1e-4, rtol=1e-4)


def test_cache_reset_continues_generation():
    """Ensure ``InferenceParams.reset`` clears state for extended generation."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = MambaConfig(d_model=64, n_layer=2, vocab_size=100)
    model = MambaLMHeadModel(cfg, device=device)

    # Use a small cache window then exceed it
    inference_params = InferenceParams(max_seqlen=3, max_batch_size=1)
    tokens = torch.randint(0, cfg.vocab_size, (1, 5), device=device)

    for i in range(inference_params.max_seqlen):
        model(tokens[:, i : i + 1], inference_params=inference_params, num_last_tokens=1)

    with pytest.raises(AssertionError):
        model(tokens[:, inference_params.max_seqlen : inference_params.max_seqlen + 1],
              inference_params=inference_params,
              num_last_tokens=1)

    # Reset with a larger window and run the full sequence again
    inference_params.reset(max_seqlen=tokens.shape[1], max_batch_size=1)
    for i in range(tokens.shape[1]):
        model(tokens[:, i : i + 1], inference_params=inference_params, num_last_tokens=1)
