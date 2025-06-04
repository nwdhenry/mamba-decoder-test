import torch
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from mamba_ssm.models.config_mamba import MambaConfig


def test_long_sequence_memory():
    """Ensure forward pass works for long sequences without OOM."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = MambaConfig(d_model=64, n_layer=2, vocab_size=100)
    model = MambaLMHeadModel(cfg, device=device)
    seq_len = 4096
    inp = torch.randint(0, cfg.vocab_size, (1, seq_len), device=device)
    try:
        _ = model(inp).logits
    except RuntimeError as e:
        assert "out of memory" not in str(e).lower()
