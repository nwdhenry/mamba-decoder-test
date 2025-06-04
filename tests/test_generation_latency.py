import time
import torch
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from mamba_ssm.models.config_mamba import MambaConfig


def test_generation_latency():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = MambaConfig(d_model=64, n_layer=2, vocab_size=100)
    model = MambaLMHeadModel(cfg, device=device)
    prompt = torch.randint(0, cfg.vocab_size, (1, 10), device=device)
    latencies = []
    for length in [50, 100, 200]:
        start = time.time()
        model.generate(input_ids=prompt, max_length=length, cg=False)
        latencies.append(time.time() - start)
    assert latencies[0] <= latencies[1] <= latencies[2]
