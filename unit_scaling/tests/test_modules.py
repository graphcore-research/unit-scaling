# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

import pytest
import torch
from torch import randint, randn

from .._modules import (
    GELU,
    MHSA,
    MLP,
    Conv1d,
    CrossEntropyLoss,
    DepthModuleList,
    DepthSequential,
    Dropout,
    Embedding,
    LayerNorm,
    Linear,
    LinearReadout,
    RMSNorm,
    SiLU,
    Softmax,
    TransformerDecoder,
    TransformerLayer,
)
from ..optim import SGD
from ..parameter import has_parameter_data
from .helper import (
    assert_non_zeros,
    assert_not_unit_scaled,
    assert_unit_scaled,
    assert_zeros,
    unit_backward,
)


def test_gelu() -> None:
    input = randn(2**10)
    model = GELU()
    output = model(input)

    assert float(output.std()) == pytest.approx(1, abs=0.1)


def test_silu() -> None:
    input = randn(2**10)
    model = SiLU()
    output = model(input)

    assert float(output.std()) == pytest.approx(1, abs=0.1)


def test_softmax() -> None:
    input = randn(2**14)
    model = Softmax(dim=0)
    output = model(input)

    # The approximation is quite rough at mult=1
    assert 0.1 < float(output.std()) < 10


def test_dropout() -> None:
    input = randn(2**12, requires_grad=True)
    model = Dropout()
    output = model(input)

    assert float(output.std()) == pytest.approx(1, abs=0.1)

    with pytest.raises(ValueError):
        Dropout(0.5, inplace=True)


def test_linear() -> None:
    input = randn(2**8, 2**10, requires_grad=True)
    model = Linear(2**10, 2**12, bias=True)
    output = model(input)

    assert_unit_scaled(model.weight)
    assert_zeros(model.bias)
    assert output.shape == torch.Size([2**8, 2**12])

    unit_backward(output)
    SGD(model.parameters(), lr=1, readout_constraint="to_output_scale").step()

    assert float(output.std()) == pytest.approx(1, abs=0.1)

    assert_not_unit_scaled(model.weight)
    assert_non_zeros(model.bias)


def test_conv1d() -> None:
    batch_size = 2**6
    d_in = 2**6 * 3
    d_out = 2**6 * 5
    kernel_size = 11
    seq_len = 2**6 * 7
    input = randn(batch_size, d_in, seq_len, requires_grad=True)
    model = Conv1d(d_in, d_out, kernel_size, bias=True)
    output = model(input)

    assert_unit_scaled(model.weight)
    assert_zeros(model.bias)

    unit_backward(output)
    SGD(model.parameters(), lr=1, readout_constraint="to_output_scale").step()

    assert float(output.std()) == pytest.approx(1, abs=0.1)

    assert_not_unit_scaled(model.weight)
    assert_non_zeros(model.bias)


def test_linear_readout() -> None:
    input = randn(2**8, 2**10, requires_grad=True)
    model = LinearReadout(2**10, 2**12)
    output = model(input)

    assert model.weight.mup_type == "output"  # type:ignore[attr-defined]
    assert_unit_scaled(model.weight)
    assert output.shape == torch.Size([2**8, 2**12])
    assert float(output.std()) == pytest.approx(2**-5, rel=0.1)

    unit_backward(output)
    SGD(model.parameters(), lr=1).step()
    assert_not_unit_scaled(model.weight)


def test_layer_norm() -> None:
    input = randn(2**8, 2**10, requires_grad=True)
    model = LayerNorm(2**10, elementwise_affine=True)
    output = model(input)

    assert output.shape == torch.Size([2**8, 2**10])

    unit_backward(output)
    SGD(model.parameters(), lr=1).step()

    assert_unit_scaled(output, input.grad, model.weight.grad, model.bias.grad)


def test_rms_norm() -> None:
    input = randn(2**8, 2**10, requires_grad=True)
    model = RMSNorm(2**10, elementwise_affine=True)
    output = model(input)

    assert output.shape == torch.Size([2**8, 2**10])
    assert model.weight is not None

    unit_backward(output)
    SGD(model.parameters(), lr=1).step()

    assert_unit_scaled(output, input.grad, model.weight.grad)


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

    assert_unit_scaled(
        model.linear_1.weight, model.linear_gate.weight, model.linear_2.weight
    )
    assert output.shape == torch.Size([2**8, 2**10])

    unit_backward(output)
    SGD(model.parameters(), lr=1, readout_constraint="to_output_scale").step()

    assert float(output.std()) == pytest.approx(1, abs=0.2)

    assert_unit_scaled(
        model.linear_1.weight.grad,
        model.linear_gate.weight.grad,
        model.linear_2.weight.grad,
    )

    assert_not_unit_scaled(
        model.linear_1.weight, model.linear_gate.weight, model.linear_2.weight
    )


def test_mhsa() -> None:
    batch_sz, seq_len, hidden_dim = 2**8, 2**6, 2**6
    input = randn(batch_sz, seq_len, hidden_dim, requires_grad=True)
    model = MHSA(hidden_dim, heads=8, is_causal=False, dropout_p=0.1)
    output = model(input)

    assert_unit_scaled(model.linear_qkv.weight, model.linear_o.weight)
    assert output.shape == torch.Size([batch_sz, seq_len, hidden_dim])

    unit_backward(output)
    SGD(model.parameters(), lr=1, readout_constraint="to_output_scale").step()

    assert float(output.std()) == pytest.approx(1, abs=0.5)

    assert_not_unit_scaled(model.linear_qkv.weight, model.linear_o.weight)


def test_transformer_layer() -> None:
    batch_sz, seq_len, hidden_dim, heads = 2**8, 2**6, 2**6, 8
    input = randn(batch_sz, seq_len, hidden_dim, requires_grad=True)
    model = TransformerLayer(
        hidden_dim,
        heads=heads,
        is_causal=False,
        dropout_p=0.1,
        mhsa_tau=0.1,
        mlp_tau=1.0,
    )
    output = model(input)

    assert output.shape == torch.Size([batch_sz, seq_len, hidden_dim])

    unit_backward(output)
    SGD(model.parameters(), lr=1).step()

    assert float(output.std()) == pytest.approx(1, abs=0.1)


def test_depth_module_list() -> None:
    layers = DepthModuleList([Linear(10, 10) for _ in range(5)])
    assert len(layers) == 5
    for layer in layers:
        assert layer.weight.mup_scaling_depth == 5

    with pytest.raises(ValueError):
        DepthModuleList([torch.nn.Linear(10, 10) for _ in range(5)])


def test_depth_sequential() -> None:
    model = DepthSequential(*(Linear(2**6, 2**6) for _ in range(7)))
    for param in model.parameters():
        assert has_parameter_data(param)
        assert param.mup_scaling_depth == 7

    input = randn(2**4, 2**6, requires_grad=True)
    output = model(input)
    unit_backward(output)
    assert_unit_scaled(output, input.grad)

    with pytest.raises(ValueError):
        DepthSequential(*[torch.nn.Linear(2**6, 2**6) for _ in range(7)])


def test_transformer_decoder() -> None:
    batch_size = 2**8
    seq_len = 2**6
    hidden_size = 2**6
    vocab_size = 2**12
    layers = 2
    heads = 4

    input_ids = torch.randint(low=0, high=vocab_size, size=(batch_size, seq_len))
    model = TransformerDecoder(hidden_size, vocab_size, layers, heads, dropout_p=0.1)
    loss = model.loss(input_ids)

    expected_loss = torch.tensor(vocab_size).log()
    assert expected_loss / 2 < loss.item() < expected_loss * 2

    loss.backward()  # type:ignore[no-untyped-call]
    SGD(model.parameters(), lr=1).step()

    for name, p in model.named_parameters():
        threshold = 5.0
        assert p.grad is not None
        assert 1 / threshold <= p.grad.std().detach() <= threshold, name
