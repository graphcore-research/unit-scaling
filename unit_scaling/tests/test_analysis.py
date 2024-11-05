# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

from typing import Tuple

import torch.nn.functional as F
from torch import Size, Tensor, nn, randn
from transformers import AutoTokenizer  # type: ignore[import-untyped]

from ..analysis import _create_batch, _example_seqs, example_batch, plot, visualiser
from ..transforms import track_scales


def test_example_seqs() -> None:
    batch_size, min_seq_len = 3, 1024
    seqs = _example_seqs(batch_size, min_seq_len)
    assert len(seqs) == batch_size, len(seqs)
    for s in seqs:
        assert isinstance(s, str)
        assert not s.isspace()
        assert len(s) >= min_seq_len


def test_create_batch() -> None:
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m-deduped")
    batch_size, seq_len = 3, 256
    seqs = _example_seqs(batch_size, min_seq_len=seq_len * 4)
    input_idxs, attn_mask, labels = _create_batch(tokenizer, seqs, seq_len)

    assert isinstance(input_idxs, Tensor)
    assert isinstance(attn_mask, Tensor)
    assert isinstance(labels, Tensor)
    assert input_idxs.shape == Size([batch_size, seq_len])
    assert attn_mask.shape == Size([batch_size, seq_len])
    assert labels.shape == Size([batch_size, seq_len])


def test_example_batch() -> None:
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m-deduped")
    batch_size, seq_len = 3, 256
    input_idxs, attn_mask, labels = example_batch(tokenizer, batch_size, seq_len)

    assert isinstance(input_idxs, Tensor)
    assert isinstance(attn_mask, Tensor)
    assert isinstance(labels, Tensor)
    assert input_idxs.shape == Size([batch_size, seq_len])
    assert attn_mask.shape == Size([batch_size, seq_len])
    assert labels.shape == Size([batch_size, seq_len])


def test_plot() -> None:
    class Model(nn.Module):
        def __init__(self, dim: int) -> None:
            super().__init__()
            self.dim = dim
            self.linear = nn.Linear(dim, dim // 2)

        def forward(self, x: Tensor) -> Tensor:  # pragma: no cover
            y = F.relu(x)
            z = self.linear(y)
            return z.sum()  # type: ignore[no-any-return]

    b, dim = 2**4, 2**8
    input = randn(b, dim)
    model = Model(dim)
    model = track_scales(model)
    loss = model(input)
    loss.backward()

    graph = model.scales_graph()
    axes = plot(graph, "demo", xmin=2**-20, xmax=2**10)
    assert axes


def test_visualiser() -> None:
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m-deduped")

    class Model(nn.Module):
        def __init__(self, n_embed: int, dim: int) -> None:
            super().__init__()
            self.embedding = nn.Embedding(n_embed, dim)
            self.linear = nn.Linear(dim, n_embed)

        def forward(
            self, inputs: Tensor, labels: Tensor
        ) -> Tuple[Tensor, Tensor]:  # pragma: no cover
            x = self.embedding(inputs)
            x = self.linear(x)
            loss = F.cross_entropy(x.view(-1, x.size(-1)), labels.view(-1))
            return x, loss

    axes = visualiser(
        model=Model(n_embed=tokenizer.vocab_size, dim=128),
        tokenizer=tokenizer,
        batch_size=16,
        seq_len=256,
    )
    assert axes
