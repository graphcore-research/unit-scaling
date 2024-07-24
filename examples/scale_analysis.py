# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import torch

import unit_scaling as uu
from unit_scaling.utils import analyse_module

print("=== Unit-scaled Linear ===\n")

batch_size = 2**8
hidden_size = 2**10
out_size = 2**10
input = torch.randn(batch_size, hidden_size).requires_grad_()
backward = torch.randn(batch_size, out_size)

annotated_code = analyse_module(uu.Linear(hidden_size, out_size), input, backward)
print(annotated_code)

print("=== Unit-scaled MLP ===\n")

batch_size = 2**8
hidden_size = 2**10
input = torch.randn(batch_size, hidden_size).requires_grad_()
backward = torch.randn(batch_size, hidden_size)

annotated_code = analyse_module(uu.MLP(hidden_size), input, backward)
print(annotated_code)

print("=== Unit-scaled MHSA ===\n")

batch_size = 2**8
seq_len = 2**6
hidden_size = 2**6
heads = 4
dropout_p = 0.1
input = torch.randn(batch_size, seq_len, hidden_size).requires_grad_()
backward = torch.randn(batch_size, seq_len, hidden_size)

annotated_code = analyse_module(
    uu.MHSA(hidden_size, heads, is_causal=False, dropout_p=dropout_p), input, backward
)
print(annotated_code)

print("=== Unit-scaled Transformer Layer ===\n")

batch_size = 2**8
seq_len = 2**6
hidden_size = 2**6
heads = 4
dropout_p = 0.1
input = torch.randn(batch_size, seq_len, hidden_size).requires_grad_()
backward = torch.randn(batch_size, seq_len, hidden_size)

annotated_code = analyse_module(
    uu.TransformerLayer(
        hidden_size,
        heads,
        mhsa_tau=0.1,
        mlp_tau=1.0,
        is_causal=False,
        dropout_p=dropout_p,
    ),
    input,
    backward,
)
print(annotated_code)

print("=== Unit-scaled Full Transformer Decoder ===\n")

batch_size = 2**8
seq_len = 2**6
hidden_size = 2**6
vocab_size = 2**12
layers = 2
heads = 4
dropout_p = 0.1

seq = torch.randint(low=0, high=vocab_size, size=(batch_size, seq_len + 1))
input_idxs = seq[:, :-1]
labels = torch.roll(seq, -1, 1)[:, 1:]

annotated_code = analyse_module(
    uu.TransformerDecoder(hidden_size, vocab_size, layers, heads, dropout_p),
    (input_idxs, labels),
)
print(annotated_code)
