# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

import pytest
import torch
from torch import randint
from torch.optim import SGD

from ..modules import (
    GELU,
    MHSA,
    MLP,
    CrossEntropyLoss,
    Dropout,
    Embedding,
    LayerNorm,
    Linear,
    Softmax,
    TransformerDecoder,
    TransformerLayer,
)
from .helper import (
    assert_non_zeros,
    assert_not_unit_scaled,
    assert_unit_scaled,
    assert_zeros,
    unit_backward,
    unit_normal,
)


def test_gelu() -> None:
    input = unit_normal(2**10)
    model = GELU()
    output = model(input)

    unit_backward(output)

    combined_std = output.std().detach() * input.grad.std()  # type: ignore
    assert combined_std == pytest.approx(1, abs=0.1)


def test_softmax() -> None:
    input = unit_normal(2**14)
    model = Softmax(dim=0)
    output = model(input)

    unit_backward(output)

    combined_std = output.std().detach() * input.grad.std()  # type: ignore
    assert combined_std == pytest.approx(1, abs=0.1)


def test_dropout() -> None:
    input = unit_normal(2**12)
    model = Dropout()
    output = model(input)

    unit_backward(output)

    combined_std = output.std().detach() * input.grad.std()  # type: ignore
    assert combined_std == pytest.approx(1, abs=0.1)

    with pytest.raises(ValueError):
        Dropout(0.5, inplace=True)


def test_linear() -> None:
    input = unit_normal(2**8, 2**10)
    model = Linear(2**10, 2**12)
    output = model(input)

    assert_unit_scaled(model.weight)
    assert_zeros(model.bias)
    assert output.shape == torch.Size([2**8, 2**12])

    unit_backward(output)
    SGD(model.parameters(), lr=1).step()

    combined_std = output.std().detach() * input.grad.std()  # type: ignore
    assert combined_std == pytest.approx(1, abs=0.1)

    assert_not_unit_scaled(model.weight)
    assert_non_zeros(model.bias)


def test_layer_norm() -> None:
    input = unit_normal(2**8, 2**10)
    model = LayerNorm(2**10)
    output = model(input)

    assert output.shape == torch.Size([2**8, 2**10])

    unit_backward(output)
    SGD(model.parameters(), lr=1).step()

    assert_unit_scaled(output, input.grad, model.weight.grad, model.bias.grad)


def test_embedding() -> None:
    batch_sz, seq_len, embedding_dim, num_embeddings = 2**4, 2**5, 2**6, 2**12
    input_idxs = randint(low=0, high=2**12, size=(batch_sz, seq_len))
    model = Embedding(num_embeddings, embedding_dim)
    output = model(input_idxs)

    assert output.shape == torch.Size([batch_sz, seq_len, embedding_dim])

    unit_backward(output)

    assert_unit_scaled(model.weight.grad)

    with pytest.raises(ValueError):
        Embedding(num_embeddings, embedding_dim, scale_grad_by_freq=True)
    with pytest.raises(ValueError):
        Embedding(num_embeddings, embedding_dim, sparse=True)


def test_cross_entropy_loss() -> None:
    num_tokens, vocab_sz = 2**12, 2**8
    input = unit_normal(num_tokens, vocab_sz)
    labels = randint(low=0, high=vocab_sz, size=(num_tokens,))
    model = CrossEntropyLoss()
    loss = model(input, labels)
    loss.backward()

    assert_unit_scaled(input.grad)

    with pytest.raises(ValueError):
        CrossEntropyLoss(weight=unit_normal(vocab_sz))
    with pytest.raises(ValueError):
        CrossEntropyLoss(label_smoothing=0.5)


def test_mlp() -> None:
    input = unit_normal(2**8, 2**10)
    model = MLP(2**10)
    output = model(input)

    assert_unit_scaled(model.linear_1.weight, model.linear_2.weight)
    assert_zeros(model.linear_1.bias, model.linear_2.bias)
    assert output.shape == torch.Size([2**8, 2**10])

    unit_backward(output)
    SGD(model.parameters(), lr=1).step()

    combined_std = output.std().detach() * input.grad.std()  # type: ignore
    assert combined_std == pytest.approx(1, abs=0.1)

    assert_not_unit_scaled(model.linear_1.weight, model.linear_2.weight)
    assert_non_zeros(model.linear_1.bias, model.linear_2.bias)


def test_mhsa() -> None:
    batch_sz, seq_len, hidden_dim = 2**8, 2**6, 2**6
    input = unit_normal(batch_sz, seq_len, hidden_dim)
    model = MHSA(hidden_dim, heads=8, dropout_p=0.1)
    output = model(input)

    assert_unit_scaled(model.linear_qkv.weight, model.linear_o.weight)
    assert output.shape == torch.Size([batch_sz, seq_len, hidden_dim])

    unit_backward(output)
    SGD(model.parameters(), lr=1).step()

    combined_std = output.std().detach() * input.grad.std()  # type: ignore
    assert combined_std == pytest.approx(1, abs=0.5)

    assert_not_unit_scaled(model.linear_qkv.weight, model.linear_o.weight)


def test_transformer_layer() -> None:
    batch_sz, seq_len, hidden_dim, heads = 2**8, 2**6, 2**6, 8
    input = unit_normal(batch_sz, seq_len, hidden_dim)
    model = TransformerLayer(hidden_dim, heads=heads, dropout_p=0.1)
    output = model(input)

    assert output.shape == torch.Size([batch_sz, seq_len, hidden_dim])

    unit_backward(output)
    SGD(model.parameters(), lr=1).step()

    combined_std = output.std().detach() * input.grad.std()  # type: ignore
    assert combined_std == pytest.approx(1, abs=0.1)


def test_transformer_decoder() -> None:
    batch_size = 2**8
    seq_len = 2**6
    hidden_size = 2**6
    vocab_size = 2**12
    layers = 2
    heads = 4

    input_idxs = torch.randint(low=0, high=vocab_size, size=(batch_size, seq_len))
    labels = torch.roll(input_idxs, -1, 1)
    model = TransformerDecoder(hidden_size, vocab_size, layers, heads, dropout_p=0.1)
    loss = model(input_idxs, labels)

    assert loss.shape == torch.Size([])

    loss.backward()
    SGD(model.parameters(), lr=1).step()

    for name, p in model.named_parameters():
        if "layer_norm.bias" in name:
            threshold = 20.0
        else:
            threshold = 5.0
        assert p.grad is not None
        assert 1 / threshold <= p.grad.std().detach() <= threshold, name
