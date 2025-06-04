import sys
import types
from unittest import mock

import torch

# Provide a dummy selective_scan_cuda so selective_scan_interface imports
sys.modules.setdefault('selective_scan_cuda', types.ModuleType('selective_scan_cuda'))

from mamba_ssm.ops import selective_scan_interface as ssi

def test_selective_scan_cpu_fallback():
    with mock.patch.object(ssi, 'SelectiveScanFn') as patched:
        patched.apply.side_effect = lambda *args, **kwargs: ssi.selective_scan_ref(*args, **kwargs)
        u = torch.randn(1, 1, 4)
        delta = torch.randn(1, 1, 4)
        A = torch.randn(1, 1)
        B = torch.randn(1, 1, 4)
        C = torch.randn(1, 1, 4)
        out = ssi.selective_scan_fn(u, delta, A, B, C)
        assert out.device.type == 'cpu'
