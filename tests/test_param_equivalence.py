import torch
from transformers import GPT2Config, GPT2Model
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from mamba_ssm.models.config_mamba import MambaConfig


def test_param_equivalence():
    """Compare parameter counts with a tiny GPT2 configuration."""
    cfg_mamba = MambaConfig(d_model=64, n_layer=2, vocab_size=100)
    mamba = MambaLMHeadModel(cfg_mamba)

    gpt_cfg = GPT2Config(n_embd=64, n_layer=2, n_head=8, vocab_size=100)
    gpt = GPT2Model(gpt_cfg)

    mamba_params = sum(p.numel() for p in mamba.parameters() if p.requires_grad)
    gpt_params = sum(p.numel() for p in gpt.parameters() if p.requires_grad)
    diff = abs(mamba_params - gpt_params) / gpt_params
    assert diff < 0.2
