import torch
from mamba_ssm.utils.generation import sample


def test_min_p_filtering_per_batch():
    torch.manual_seed(0)
    logits = torch.tensor([[1.0, 2.0, 3.0], [1.0, 10.0, 0.0]])
    out = sample(logits, top_k=0, top_p=0.0, min_p=0.5, temperature=1.0)
    # First batch should never select index 0 as its logit is below the threshold
    assert out[0].item() in {1, 2}
    # Second batch should only be able to select the highest logit token
    assert out[1].item() == 1
