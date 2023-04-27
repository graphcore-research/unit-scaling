# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import torch

from unit_scaling.modules import MHSA, MLP, Linear, TransformerDecoder, TransformerLayer
from unit_scaling.utils import analyse_module

print("=== Unit-scaled Linear ===\n")

batch_size = 2**8
hidden_size = 2**10
out_size = 2**10
input = torch.randn(batch_size, hidden_size).requires_grad_()
backward = torch.randn(batch_size, out_size)

annotated_code = analyse_module(Linear(hidden_size, out_size), input, backward)
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

print("=== Unit-scaled Transformer Layer ===\n")

batch_size = 2**8
seq_len = 2**6
hidden_size = 2**6
heads = 4
input = torch.randn(batch_size, seq_len, hidden_size).requires_grad_()
backward = torch.randn(batch_size, seq_len, hidden_size)

annotated_code = analyse_module(TransformerLayer(hidden_size, heads), input, backward)
print(annotated_code)

print("=== Unit-scaled Full Transformer Decoder ===\n")

batch_size = 2**8
seq_len = 2**6
hidden_size = 2**6
vocab_size = 2**12
layers = 2
heads = 4

input_idxs = torch.randint(low=0, high=vocab_size, size=(batch_size, seq_len))
labels = torch.roll(input_idxs, -1, 1)  # Note: doesn't handle final index correctly
backward = torch.randn(batch_size, seq_len, hidden_size)

annotated_code = analyse_module(
    TransformerDecoder(hidden_size, vocab_size, layers, heads), (input_idxs, labels)
)
print(annotated_code)
