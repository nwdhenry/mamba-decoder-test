from dataclasses import dataclass
import torch
from torch import nn

@dataclass
class GPTConfig:
    vocab_size: int = 65
    block_size: int = 8
    n_layer: int = 1
    n_head: int = 2
    n_embd: int = 16

class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.token_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)
        layer = nn.TransformerEncoderLayer(config.n_embd, config.n_head, 4 * config.n_embd)
        self.transformer = nn.TransformerEncoder(layer, config.n_layer)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        b, t = idx.shape
        assert t <= self.config.block_size
        pos = torch.arange(t, device=idx.device)
        x = self.token_emb(idx) + self.pos_emb(pos)
        x = self.transformer(x)
        logits = self.lm_head(x)
        return logits
