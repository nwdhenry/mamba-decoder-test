import pytest
import torch
from mamba_ssm.models.mamba_gpt import MambaGPT, MambaGPTConfig
from unittest import mock


def _estimate_vram_gb(model: MambaGPT, batch_size: int = 1, seq_len: int = 128) -> float:
    """Rudimentary VRAM estimator in GB."""
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    dtype_bytes = 4  # Assume fp32 activations
    act_bytes = batch_size * seq_len * model.config.d_model * model.config.n_layer * dtype_bytes
    if model.gradient_checkpointing:
        act_bytes *= 0.5
    total_bytes = param_bytes + act_bytes
    return total_bytes / 1024 ** 3


@pytest.mark.vram_limit(11)
def test_vram_under_cap(request):
    """Ensure memory usage stays below the VRAM cap with and without checkpointing."""
    limit = request.node.get_closest_marker("vram_limit").args[0]
    cfg = MambaGPTConfig(d_model=64, n_layer=2, vocab_size=100)
    mock_props = mock.Mock(total_memory=limit * 1024 ** 3)
    with mock.patch("torch.cuda.get_device_properties", return_value=mock_props):
        model_no_ckpt = MambaGPT(cfg, device="cpu", gradient_checkpointing=False)
        model_ckpt = MambaGPT(cfg, device="cpu", gradient_checkpointing=True)

    mem_no_ckpt = _estimate_vram_gb(model_no_ckpt)
    mem_ckpt = _estimate_vram_gb(model_ckpt)

    assert mem_ckpt < mem_no_ckpt
    assert mem_ckpt <= limit
    assert mem_no_ckpt <= limit

