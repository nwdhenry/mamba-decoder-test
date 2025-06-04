import torch
from dataclasses import dataclass
from typing import Dict

PRESET_CONFIGS: Dict[str, Dict[str, int]] = {
    "tiny": {"d_model": 256, "n_layer": 8},
    "small": {"d_model": 512, "n_layer": 12},
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

def autotune_batch_size(free_mem_gb: float) -> int:
    if free_mem_gb < 4:
        return 1
    elif free_mem_gb < 8:
        return 2
    elif free_mem_gb < 16:
        return 4
    else:
        return 8

@dataclass
class ContextLenWarmup:
    start: int = 512
    target: int = 2048
    steps: int = 1000

    def get(self, step: int) -> int:
        if step >= self.steps:
            return self.target
        return int(self.start + (self.target - self.start) * step / self.steps)
