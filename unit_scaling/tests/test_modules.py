# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

import pytest
import torch
from torch import randint, randn
from torch.optim import SGD

from .._modules import (
    GELU,
    MHSA,
    MLP,
    CrossEntropyLoss,
    Dropout,
    Embedding,
    LayerNorm,
    Linear,
    Softmax,
    TransformerLayer,
)
from .helper import (
    assert_non_zeros,
    assert_not_unit_scaled,
    assert_scale,
    assert_unit_scaled,
    assert_zeros,
    unit_backward,
)


def test_gelu() -> None:
    input = randn(2**10, requires_grad=True)
    model = GELU()
    output = model(input)

    unit_backward(output)

    combined_std = output.std().detach() * input.grad.std()  # type: ignore
    assert combined_std == pytest.approx(1, abs=0.1)


def test_softmax() -> None:
    input = randn(2**14, requires_grad=True)
    model = Softmax(dim=0)
    output = model(input)

    unit_backward(output)

    combined_std = output.std().detach() * input.grad.std()  # type: ignore
    assert combined_std == pytest.approx(1, abs=0.1)


def test_dropout() -> None:
    input = randn(2**12, requires_grad=True)
    model = Dropout()
    output = model(input)

    unit_backward(output)

    combined_std = output.std().detach() * input.grad.std()  # type: ignore
    assert combined_std == pytest.approx(1, abs=0.1)

    with pytest.raises(ValueError):
        Dropout(0.5, inplace=True)


def test_linear() -> None:
    b = 2**8
    input = randn(b, 2**10, requires_grad=True)
    model = Linear(2**10, 2**12)
    output = model(input)

    assert_unit_scaled(model.weight)
    assert_zeros(model.bias)
    assert output.shape == torch.Size([2**8, 2**12])

    unit_backward(output)
    SGD(model.parameters(), lr=10).step()

    combined_std = output.std().detach() * input.grad.std()  # type: ignore
    assert combined_std == pytest.approx(1, abs=0.1)

    assert_not_unit_scaled(model.weight)
    assert_non_zeros(model.bias)


def test_layer_norm() -> None:
    b = 2**8
    input = randn(b, 2**10, requires_grad=True)
    model = LayerNorm(2**10)
    output = model(input)

    assert output.shape == torch.Size([b, 2**10])

    unit_backward(output)
    SGD(model.parameters(), lr=1).step()

    assert_unit_scaled(output, input.grad)
    assert_scale(model.weight.grad, model.bias.grad, target=b**-0.25)


def test_embedding() -> None:
    batch_sz, seq_len, embedding_dim, num_embeddings = 2**4, 2**5, 2**6, 2**12
    input_idxs = randint(low=0, high=2**12, size=(batch_sz, seq_len))
    model = Embedding(num_embeddings, embedding_dim)
    output = model(input_idxs)

    assert output.shape == torch.Size([batch_sz, seq_len, embedding_dim])

    unit_backward(output)

    assert_scale(
        model.weight.grad, target=(num_embeddings / (batch_sz * seq_len)) ** 0.25
    )

    with pytest.raises(ValueError):
        Embedding(num_embeddings, embedding_dim, scale_grad_by_freq=True)
    with pytest.raises(ValueError):
        Embedding(num_embeddings, embedding_dim, sparse=True)


def test_cross_entropy_loss() -> None:
    num_tokens, vocab_sz = 2**12, 2**8
    input = randn(num_tokens, vocab_sz, requires_grad=True)
    labels = randint(low=0, high=vocab_sz, size=(num_tokens,))
    model = CrossEntropyLoss()
    loss = model(input, labels)
    loss.backward()

    assert_unit_scaled(input.grad)

    with pytest.raises(ValueError):
        CrossEntropyLoss(weight=randn(vocab_sz))
    with pytest.raises(ValueError):
        CrossEntropyLoss(label_smoothing=0.5)


def test_mlp() -> None:
    input = randn(2**8, 2**10, requires_grad=True)
    model = MLP(2**10)
    output = model(input)

    assert_unit_scaled(model.linear_1.weight, model.linear_2.weight)
    assert_zeros(model.linear_1.bias, model.linear_2.bias)
    assert output.shape == torch.Size([2**8, 2**10])

    unit_backward(output)
    SGD(model.parameters(), lr=10).step()

    combined_std = output.std().detach() * input.grad.std()  # type: ignore
    assert combined_std == pytest.approx(1, abs=0.1)

    assert_not_unit_scaled(model.linear_1.weight, model.linear_2.weight)
    assert_non_zeros(model.linear_1.bias, model.linear_2.bias)


def test_mhsa() -> None:
    batch_sz, seq_len, hidden_dim = 2**8, 2**6, 2**6
    input = randn(batch_sz, seq_len, hidden_dim, requires_grad=True)
    model = MHSA(hidden_dim, heads=8, dropout_p=0.1)
    output = model(input)

    assert_unit_scaled(model.linear_qkv.weight, model.linear_o.weight)
    assert output.shape == torch.Size([batch_sz, seq_len, hidden_dim])

    unit_backward(output)
    SGD(model.parameters(), lr=10).step()

    combined_std = output.std().detach() * input.grad.std()  # type: ignore
    assert combined_std == pytest.approx(1, abs=0.5)

    assert_not_unit_scaled(model.linear_qkv.weight, model.linear_o.weight)


def test_transformer_layer() -> None:
    batch_sz, seq_len, hidden_dim, heads = 2**8, 2**6, 2**6, 8
    input = randn(batch_sz, seq_len, hidden_dim, requires_grad=True)
    model = TransformerLayer(hidden_dim, heads=heads, dropout_p=0.1)
    output = model(input)

    assert output.shape == torch.Size([batch_sz, seq_len, hidden_dim])

    unit_backward(output)
    SGD(model.parameters(), lr=10).step()

    combined_std = output.std().detach() * input.grad.std()  # type: ignore
    assert combined_std == pytest.approx(1, abs=0.1)
