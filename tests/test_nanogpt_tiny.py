import torch
import pytest

nanoGPT = pytest.importorskip("nanoGPT")
from nanoGPT import GPTConfig, GPT


def test_nanogpt_tiny_forward():
    config = GPTConfig()
    model = GPT(config)
    x = torch.randint(0, config.vocab_size, (2, config.block_size))
    out = model(x)
    assert out.shape == (2, config.block_size, config.vocab_size)
