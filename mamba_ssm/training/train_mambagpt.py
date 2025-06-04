import argparse
from contextlib import nullcontext
from functools import partial

import torch
from torch.utils.data import IterableDataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import torch.nn.functional as F

from transformers import AutoTokenizer
from bitsandbytes.optim import Adam8bit

from mamba_ssm.models.mamba_gpt import MambaGPT, MambaGPTConfig
from mamba_ssm.training.autoconfig import (
    PRESET_CONFIGS,
    ContextLenWarmup,
    autotune_batch_size,
    detect_total_vram_gb,
    select_preset_by_vram,
)


class StreamingTextDataset(IterableDataset):
    """Stream text from a file and tokenize on the fly."""

    def __init__(self, path: str, tokenizer, seq_len: int = 128):
        self.path = path
        self.tokenizer = tokenizer
        self.seq_len = seq_len

    def __iter__(self):
        buffer = []
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                tokens = self.tokenizer.encode(line.strip(), add_special_tokens=False)
                buffer.extend(tokens)
                while len(buffer) > self.seq_len:
                    x = buffer[: self.seq_len]
                    y = buffer[1 : self.seq_len + 1]
                    buffer = buffer[self.seq_len :]
                    yield torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


def monitor_vram(threshold_gb: float = 11.0):
    reserved = torch.cuda.memory_reserved() / 1024 ** 3
    if reserved > threshold_gb:
        print(f"[WARNING] VRAM usage {reserved:.2f}GB exceeds {threshold_gb}GB")


def main():
    parser = argparse.ArgumentParser(description="Train MambaGPT with low VRAM usage")
    parser.add_argument("--train-file", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=0, help="0 to auto tune")
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--checkpointing", action="store_true")
    parser.add_argument("--preset", type=str, choices=list(PRESET_CONFIGS.keys()))
    parser.add_argument("--auto-config", action="store_true", help="select preset based on VRAM")
    parser.add_argument("--warmup-steps", type=int, default=1000)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp_dtype = torch.float32
    if args.bf16:
        amp_dtype = torch.bfloat16
    elif args.fp16:
        amp_dtype = torch.float16

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")

    total_vram = detect_total_vram_gb()
    if args.auto_config and args.preset is None:
        args.preset = select_preset_by_vram(total_vram)
    cfg_kwargs = PRESET_CONFIGS.get(args.preset or "base", {})
    config = MambaGPTConfig(**cfg_kwargs)
    model = MambaGPT(config, device=device, gradient_checkpointing=args.checkpointing)
    model.to(device)

    if args.batch_size == 0:
        args.batch_size = autotune_batch_size()

    warmup = ContextLenWarmup(target=args.seq_len, steps=args.warmup_steps)
    ds = StreamingTextDataset(args.train_file, tokenizer, seq_len=warmup.start)
    loader = DataLoader(ds, batch_size=args.batch_size)

    
    optimizer = Adam8bit(model.parameters(), lr=args.lr)
    scaler = GradScaler(enabled=amp_dtype != torch.float32)

    model.train()
    step = 0
    for _ in range(args.epochs):
        for input_ids, targets in loader:
            ds.seq_len = warmup.get(step)
            input_ids = input_ids.to(device)
            targets = targets.to(device)
            try:
                with autocast(device_type="cuda", dtype=amp_dtype) if device == "cuda" else nullcontext():
                    logits = model(input_ids)
                    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                monitor_vram()
                print(f"loss: {loss.item():.4f}")
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    torch.cuda.empty_cache()
                    if args.batch_size > 1:
                        args.batch_size = max(1, args.batch_size // 2)
                        loader = DataLoader(ds, batch_size=args.batch_size)
                        print(f"[WARN] OOM encountered. Reducing batch size to {args.batch_size}")
                        continue
                    else:
                        print("[ERROR] OOM with batch size 1. Skipping batch.")
                        continue
                else:
                    raise
            step += 1


if __name__ == "__main__":
    main()
