import torch
from torch.cuda.amp import autocast, GradScaler
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from mamba_ssm.models.config_mamba import MambaConfig


def test_amp_training_step():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = MambaConfig(d_model=64, n_layer=2, vocab_size=100)
    model = MambaLMHeadModel(cfg, device=device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    scaler = GradScaler(enabled=True)
    inp = torch.randint(0, cfg.vocab_size, (1, 32), device=device)
    tgt = inp.clone()
    with autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", dtype=torch.bfloat16):
        logits = model(inp).logits
        loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), tgt.view(-1))
    scaler.scale(loss).backward()
    scaler.step(opt)
    scaler.update()
    assert not torch.isnan(loss).any()
