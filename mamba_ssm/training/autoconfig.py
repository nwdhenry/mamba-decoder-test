import torch
from dataclasses import dataclass
from typing import Dict

# Preset configurations used to quickly instantiate common model sizes.
# These values are referenced by the training scripts and some utilities.
PRESET_CONFIGS: Dict[str, Dict[str, int]] = {
    "tiny": {"d_model": 256, "n_layer": 8},
    "small": {"d_model": 512, "n_layer": 10},
    "base": {"d_model": 768, "n_layer": 12},
}

def detect_total_vram_gb(device=None) -> float:
    if not torch.cuda.is_available():
        return 0.0
    if device is None:
        device = torch.cuda.current_device()
    prop = torch.cuda.get_device_properties(device)
    return prop.total_memory / 1024 ** 3

def detect_free_vram_gb(device=None) -> float:
    if not torch.cuda.is_available():
        return 0.0
    if device is None:
        device = torch.cuda.current_device()
    free, _ = torch.cuda.mem_get_info(device)
    return free / 1024 ** 3

def select_preset_by_vram(total_vram_gb: float) -> str:
    if total_vram_gb < 10:
        return "tiny"
    elif total_vram_gb < 20:
        return "small"
    else:
        return "base"

def est_batch_vram(batch_size: int) -> float:
    """Return a rough estimate of VRAM usage in GB for ``batch_size``.

    The heuristic assumes ~2GB of memory is required per sample at the
    default sequence length.  This should be adjusted for different
    model sizes or training settings.
    """

    return batch_size * 2.0


def autotune_batch_size(buffer: float = 1.0, max_batch_size: int = 8) -> int:
    """Heuristically select a batch size that fits in available VRAM.

    Starting from ``1``, the batch size is doubled while the detected
    free VRAM minus ``buffer`` is larger than the estimated footprint of
    the current batch. ``est_batch_vram`` assumes roughly 2GB per sample.
    """

    batch_size = 1
    while batch_size < max_batch_size:
        free_mem = detect_free_vram_gb()
        if free_mem - buffer > est_batch_vram(batch_size):
            batch_size *= 2
        else:
            break
    return min(batch_size, max_batch_size)

@dataclass
class ContextLenWarmup:
    start: int = 512
    target: int = 2048
    steps: int = 10_000

    def get(self, step: int) -> int:
        if step >= self.steps:
            return self.target
        return int(self.start + (self.target - self.start) * step / self.steps)
