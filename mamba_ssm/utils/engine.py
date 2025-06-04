import time
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import torch
from torch import Tensor
from transformers.generation import TextStreamer

from .generation import InferenceParams, sample


@dataclass
class GenerationRequest:
    """Container holding information for a single generation request."""

    input_ids: Tensor
    max_length: int
    top_k: int = 1
    top_p: float = 0.0
    temperature: float = 1.0
    eos_token_id: Optional[int] = None
    streamer: Optional[TextStreamer] = None
    generated: List[int] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)


class LatencyMonitor:
    """Simple latency tracker with percentile statistics."""

    def __init__(self) -> None:
        self._durations: List[float] = []

    def record(self, duration: float) -> None:
        self._durations.append(duration)

    def percentiles(self, pct: Sequence[int] = (50, 90, 95, 99)) -> dict:
        if not self._durations:
            return {p: 0.0 for p in pct}
        d = torch.tensor(self._durations)
        return {p: d.quantile(p / 100.0).item() for p in pct}


class ContinuousBatchGenerationEngine:
    """Naïve engine that batches requests for autoregressive generation."""

    def __init__(self, model) -> None:
        self.model = model
        self.monitor = LatencyMonitor()

    @torch.inference_mode()
    def generate(self, requests: List[GenerationRequest]) -> List[Tensor]:
        if len(requests) == 0:
            return []
        device = next(iter(self.model.parameters())).device
        batch_size = len(requests)
        input_lens = [req.input_ids.shape[1] for req in requests]
        max_len = max(req.max_length for req in requests)

        # Prepare concatenated prompts for varlen prefix pass
        concat_inputs = torch.cat([r.input_ids.to(device) for r in requests], dim=1)
        seq_idx = torch.cat(
            [torch.full((l,), i, dtype=torch.int32, device=device) for i, l in enumerate(input_lens)],
            dim=0,
        ).unsqueeze(0)
        cu_seqlens = torch.tensor(input_lens, device=device, dtype=torch.int32).cumsum(dim=0)
        cu_seqlens = torch.nn.functional.pad(cu_seqlens, (1, 0))

        inference_params = InferenceParams(max_seqlen=max_len, max_batch_size=batch_size)
        logits = self.model(
            concat_inputs,
            inference_params=inference_params,
            seq_idx=seq_idx,
            cu_seqlens=cu_seqlens,
        ).logits
        logits = logits[0, cu_seqlens[1:] - 1]

        next_tokens = []
        for i, req in enumerate(requests):
            if req.streamer is not None:
                req.streamer.put(req.input_ids.cpu())
            token = sample(
                logits[i : i + 1],
                top_k=req.top_k,
                top_p=req.top_p,
                temperature=req.temperature,
            )
            req.generated.append(token.item())
            next_tokens.append(token)
        tokens = torch.stack(next_tokens, dim=0).unsqueeze(1)

        sequences = [torch.cat([req.input_ids.to(device), tokens[i : i + 1]], dim=1) for i, req in enumerate(requests)]
        finished = [False] * batch_size

        for i, (req, t) in enumerate(zip(requests, next_tokens)):
            if req.streamer is not None:
                req.streamer.put(t.cpu())
            if req.eos_token_id is not None and t.item() == req.eos_token_id:
                finished[i] = True
                if req.streamer is not None:
                    req.streamer.end()
                self.monitor.record(time.time() - req.start_time)

        while not all(finished):
            inference_params.seqlen_offset += 1
            logits = self.model(tokens, inference_params=inference_params, num_last_tokens=1).logits
            next_tokens = []
            for i, req in enumerate(requests):
                if finished[i]:
                    next_tokens.append(tokens[i])
                    continue
                token = sample(
                    logits[i : i + 1],
                    top_k=req.top_k,
                    top_p=req.top_p,
                    temperature=req.temperature,
                )
                req.generated.append(token.item())
                sequences[i] = torch.cat([sequences[i], token.unsqueeze(0).to(device)], dim=1)
                if req.streamer is not None:
                    req.streamer.put(token.cpu())
                if (
                    req.eos_token_id is not None and token.item() == req.eos_token_id
                ) or sequences[i].shape[1] >= req.max_length:
                    finished[i] = True
                    if req.streamer is not None:
                        req.streamer.end()
                    self.monitor.record(time.time() - req.start_time)
                next_tokens.append(token)
            tokens = torch.stack(next_tokens, dim=0).unsqueeze(1)
        return [seq.cpu() for seq in sequences]

