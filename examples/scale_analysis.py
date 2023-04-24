import torch

from unit_scaling.modules import MHSA, MLP, Linear
from unit_scaling.utils import analyse_module

print("=== Unit-scaled Linear ===\n")

batch_size = 2**8
hidden_size = 2**10
out_size = 2**10
input = torch.randn(batch_size, hidden_size).requires_grad_()
backward = torch.randn(batch_size, out_size)

annotated_code = analyse_module(
    Linear(hidden_size, out_size, bias=False), input, backward
)
print(annotated_code)

print("=== Unit-scaled MLP ===\n")

batch_size = 2**8
hidden_size = 2**10
input = torch.randn(batch_size, hidden_size).requires_grad_()
backward = torch.randn(batch_size, hidden_size)

annotated_code = analyse_module(MLP(hidden_size), input, backward)
print(annotated_code)

print("=== Unit-scaled MHSA ===\n")

batch_size = 2**8
seq_len = 2**6
hidden_size = 2**6
heads = 4
input = torch.randn(batch_size, seq_len, hidden_size).requires_grad_()
backward = torch.randn(batch_size, seq_len, hidden_size)

annotated_code = analyse_module(MHSA(hidden_size, heads), input, backward)
print(annotated_code)
