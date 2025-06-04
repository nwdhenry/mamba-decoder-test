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
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--checkpointing", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp_dtype = torch.float32
    if args.bf16:
        amp_dtype = torch.bfloat16
    elif args.fp16:
        amp_dtype = torch.float16

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    ds = StreamingTextDataset(args.train_file, tokenizer, seq_len=args.seq_len)
    loader = DataLoader(ds, batch_size=args.batch_size)

    config = MambaGPTConfig()
    model = MambaGPT(config, device=device, gradient_checkpointing=args.checkpointing)
    model.to(device)

    optimizer = Adam8bit(model.parameters(), lr=args.lr)
    scaler = GradScaler(enabled=amp_dtype != torch.float32)

    model.train()
    for _ in range(args.epochs):
        for input_ids, targets in loader:
            input_ids = input_ids.to(device)
            targets = targets.to(device)
            with autocast(device_type="cuda", dtype=amp_dtype) if device == "cuda" else nullcontext():
                logits = model(input_ids)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            monitor_vram()
            print(f"loss: {loss.item():.4f}")


if __name__ == "__main__":
    main()
