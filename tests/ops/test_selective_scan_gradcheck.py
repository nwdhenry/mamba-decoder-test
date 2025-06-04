import pytest

# Skip if torch not available
torch = pytest.importorskip("torch")

# Skip if CUDA unavailable
pytest.skip("CUDA required for gradcheck", allow_module_level=True) if not torch.cuda.is_available() else None

from mamba_ssm.ops.selective_scan_interface import selective_scan_fn


def _run_gradcheck():
    device = 'cuda'
    dtype = torch.double
    batch, dim, seqlen, dstate = 1, 2, 4, 3
    u = torch.randn(batch, dim, seqlen, dtype=dtype, device=device, requires_grad=True)
    delta = torch.randn(batch, dim, seqlen, dtype=dtype, device=device, requires_grad=True)
    A = torch.randn(dim, dstate, dtype=dtype, device=device, requires_grad=True)
    B = torch.randn(dim, dstate, dtype=dtype, device=device, requires_grad=True)
    C = torch.randn(dim, dstate, dtype=dtype, device=device, requires_grad=True)

    def func(u, delta, A, B, C):
        return selective_scan_fn(u, delta, A, B, C, delta_softplus=True)

    return torch.autograd.gradcheck(func, (u, delta, A, B, C))


def test_selective_scan_gradcheck():
    assert _run_gradcheck()
