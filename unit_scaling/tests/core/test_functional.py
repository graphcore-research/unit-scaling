# Copyright (c) 2024 Graphcore Ltd. All rights reserved.

import pytest
from torch import ones, randn, tensor
from torch.testing import assert_close

from ...core.functional import rms, scale_elementwise, transformer_residual_scaling_rule
from ..helper import unit_backward


def test_scale_elementwise_no_constraint() -> None:
    input = randn(2**10, requires_grad=True)
    f = lambda x: x
    scaled_f = scale_elementwise(
        f, output_scale=2.5, grad_input_scale=0.5, constraint=None
    )
    output = scaled_f(input)
    unit_backward(output)

    assert output.std().detach() == pytest.approx(2.5, rel=0.1)
    assert input.grad.std().detach() == pytest.approx(0.5, rel=0.1)  # type: ignore


def test_scale_elementwise_for_output() -> None:
    input = randn(2**10, requires_grad=True)
    f = lambda x: x
    scaled_f = scale_elementwise(
        f, output_scale=2.5, grad_input_scale=0.5, constraint="to_output_scale"
    )
    output = scaled_f(input)
    unit_backward(output)

    assert output.std().detach() == pytest.approx(2.5, rel=0.1)
    assert input.grad.std().detach() == pytest.approx(2.5, rel=0.1)  # type: ignore


def test_scale_elementwise_for_grad_input() -> None:
    input = randn(2**10, requires_grad=True)
    f = lambda x: x
    scaled_f = scale_elementwise(
        f, output_scale=2.5, grad_input_scale=0.5, constraint="to_grad_input_scale"
    )
    output = scaled_f(input)
    unit_backward(output)

    assert output.std().detach() == pytest.approx(0.5, rel=0.1)
    assert input.grad.std().detach() == pytest.approx(0.5, rel=0.1)  # type: ignore


def test_rms() -> None:
    output = rms(-4 + 3 * randn(2**12))
    assert output.item() == pytest.approx(5, rel=0.1)

    x = tensor([[2, -2, -2, -2], [0, 2, 0, 0]]).float()
    output = rms(x, dims=(1,))
    assert_close(output, tensor([2.0, 1.0]))

    output = rms(tensor([0.0, 0.0, 0.0]), eps=1 / 16)
    assert output.item() == pytest.approx(1 / 4, rel=0.1)


@pytest.mark.parametrize(
    ["residual_mult", "residual_attn_ratio", "layers"],
    [
        (1.0, 1.0, 4),
        (0.5, 1.0, 4),
        (1.0, 2.0, 4),
        (0.5, 1 / 3, 6),
    ],
)
def test_transformer_residual_scaling_rule(
    residual_mult: float, residual_attn_ratio: float, layers: int
) -> None:
    scaling_rule = transformer_residual_scaling_rule(
        residual_mult=residual_mult,
        residual_attn_ratio=residual_attn_ratio,
    )
    scales = ones(layers + 1)
    for n in range(layers):
        tau = scaling_rule(n, layers)
        scales[n + 1] *= tau
        scales[: n + 2] /= (1 + tau**2) ** 0.5
    embedding_scale = scales[0]
    attn_scales = scales[1:][::2]
    mlp_scales = scales[2:][::2]

    # Basic properties
    assert_close(scales.pow(2).sum(), tensor(1.0))

    s_embedding = embedding_scale
    s_attn = attn_scales.pow(2).sum().sqrt()
    s_mlp = mlp_scales.pow(2).sum().sqrt()
    s_attn_mlp_average = ((s_attn**2 + s_mlp**2) / 2).sqrt()

    assert_close(s_attn_mlp_average / s_embedding, tensor(residual_mult))
    assert_close(s_attn / s_mlp, tensor(residual_attn_ratio))

    # Per-layer scales are equal
    assert_close(attn_scales, attn_scales[:1].broadcast_to(attn_scales.shape))
    assert_close(mlp_scales, mlp_scales[:1].broadcast_to(mlp_scales.shape))
