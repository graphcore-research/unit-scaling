# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

from itertools import islice
import math
from pathlib import Path
from typing import *

import einops
import torch
from torch import nn, Tensor
import tqdm


class Config(dict):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.__dict__ = self


CONFIG = Config(
    sequence_length=256,
    batch_size=16,
    hidden_size=256,
    head_size=64,
    depth=4,
    fully_scaled_attention=False,
    lr=2**-10,
    steps=5000,
)


# https://www.gutenberg.org/cache/epub/100/pg100.txt
DATA = torch.tensor(list(Path("shakespeare.txt").read_bytes()))


def batches() -> Iterable[Tensor]:
    while True:
        offsets = torch.randint(
            len(DATA) - CONFIG.sequence_length - 1, (CONFIG.batch_size,)
        )
        yield torch.stack([DATA[i : i + CONFIG.sequence_length + 1] for i in offsets])


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.head_size = CONFIG.head_size
        self.n_heads = CONFIG.hidden_size // CONFIG.head_size
        self.qkv = nn.Linear(CONFIG.hidden_size, 3 * self.n_heads * self.head_size)
        self.proj = nn.Linear(self.n_heads * self.head_size, CONFIG.hidden_size)
        # Put the scale in a non-trainable parameter, to avoid recompilation
        self.out_scale = nn.Parameter(
            torch.tensor(
                (CONFIG.sequence_length / math.e) ** 0.5
                if CONFIG.fully_scaled_attention
                else 1.0
            ),
            requires_grad=False,
        )

    def forward(self, x: Tensor) -> Tensor:
        s = x.shape[1]
        q, k, v = einops.rearrange(
            self.qkv(x), "b s (M n d) -> M b n s d", M=3, n=self.n_heads
        )
        qk_scale = torch.tensor(self.head_size**-0.5, dtype=x.dtype, device=x.device)
        pre_a = torch.einsum("bnsd, bntd -> bnst", q, k) * qk_scale
        pre_a = pre_a + torch.triu(
            torch.full((s, s), -1e4, device=x.device), diagonal=1
        )
        a = torch.softmax(pre_a, -1)
        out = torch.einsum("bnst, bntd -> bnsd", a, v) * self.out_scale
        return self.proj(einops.rearrange(out, "b n s d -> b s (n d)"))


class FFN(nn.Module):
    def __init__(self):
        super().__init__()
        self.up = nn.Linear(CONFIG.hidden_size, 4 * CONFIG.hidden_size)
        self.down = nn.Linear(self.up.out_features, self.up.in_features)

    def forward(self, x: Tensor) -> Tensor:
        return self.down(torch.nn.functional.gelu(self.up(x)))


class PreNormResidual(nn.Module):
    def __init__(self, body: nn.Module):
        super().__init__()
        self.norm = nn.LayerNorm([CONFIG.hidden_size])
        self.body = body

    def forward(self, x: Tensor) -> Tensor:
        return x + self.body(self.norm(x))


class AbsolutePositionalEncoding(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(
            torch.randn(CONFIG.sequence_length, CONFIG.hidden_size)
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.weight


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Embedding(256, CONFIG.hidden_size),
            AbsolutePositionalEncoding(),
            nn.LayerNorm([CONFIG.hidden_size]),
            *(
                nn.Sequential(PreNormResidual(Attention()), PreNormResidual(FFN()))
                for _ in range(CONFIG.depth)
            ),
            nn.LayerNorm([CONFIG.hidden_size]),
            nn.Linear(CONFIG.hidden_size, 256),
        )

    def forward(self, indices: Tensor) -> Tensor:
        return nn.functional.cross_entropy(
            self.model(indices[:, :-1]).flatten(0, -2), indices[:, 1:].flatten()
        )


def train() -> Tensor:
    model = Model()
    opt = torch.optim.Adam(model.parameters(), lr=CONFIG.lr)
    losses = []
    for batch in tqdm.tqdm(islice(batches(), CONFIG.steps), total=CONFIG.steps):
        opt.zero_grad()
        loss = model(batch)
        loss.backward()
        opt.step()
        losses.append(float(loss))
    return torch.tensor(losses)
