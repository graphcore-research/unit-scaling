# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

import pickle
import time
from itertools import islice
from pathlib import Path
from typing import *

import numpy as np
import poptorch
import torch
import tqdm
from torch import Tensor, nn
from torch.fx.graph_module import GraphModule

import nanoGPT.model
import unit_scaling.transforms.utils
from unit_scaling.formats import FPFormat

# Basic standard config

DATA_PATH = Path(nanoGPT.model.__file__).parent / "data/shakespeare_char"
DATA = {
    split: torch.frombuffer(
        (DATA_PATH / f"{split}.bin").read_bytes(), dtype=torch.int16
    )
    for split in ["train", "val"]
}
META = pickle.loads((DATA_PATH / "meta.pkl").read_bytes())

CONFIG = nanoGPT.model.GPTConfig(
    vocab_size=META["vocab_size"],
    block_size=256,
    n_layer=6,
    n_head=6,
    n_embd=384,
    dropout=0.2,
    bias=True,  # NOTE: bias=False doesn't work
)
BATCH_SIZE = 64
MAX_ITERS = 5000
WARMUP_ITERS = 100
LR = 1e-3
MIN_LR = 1e-4
EVAL_INTERVAL = 250
EVAL_ITERS = 200

# Added settings
COMPUTE_BATCH_SIZE = 4
REPLICATION_FACTOR = 4

assert BATCH_SIZE % (COMPUTE_BATCH_SIZE * REPLICATION_FACTOR) == 0


def batches(split: str) -> Iterable[Tuple[Tensor, Tensor]]:
    d = DATA[split]
    while True:
        idx = torch.randint(len(d) - CONFIG.block_size, (BATCH_SIZE,))
        tokens = torch.stack([d[i : i + CONFIG.block_size] for i in idx]).to(torch.long)
        yield tokens[:, :-1].contiguous(), tokens[:, 1:].contiguous()


def transform(model: nanoGPT.model.GPT, unit_scaled: bool, fake_fp8: bool) -> nn.Module:
    """Add unit scaling and/or fake FP8 graph transforms, in an IPU-friendly manner.

    Note that this modifies the original model.
    """

    # This override prevents the use of a device="cpu" arange() tensor, which breaks
    # compilation, since `embed(arange(n))`` is just the same as `embed.weight[:n]`
    def _wpe_arange(self: nn.Embedding, input: Tensor) -> Tensor:
        return self.weight[: input.shape[0]]

    model.transformer.wpe.forward = _wpe_arange.__get__(
        model.transformer.wpe, model.transformer.wpe.__class__
    )

    if unit_scaled:
        unit_scaling.transforms._unit_scale._unit_init_weights(model)
        unit_scaling.transforms._unit_scale._zero_init_biases(model)

    graph_module: GraphModule

    def _backend(gm: GraphModule, example_inputs: List[Tensor]) -> GraphModule:
        if unit_scaled:
            gm = unit_scaling.transforms._unit_scale.unit_scaling_backend()(
                gm, example_inputs
            )
        if fake_fp8:
            gm = unit_scaling.transforms._simulate_format._quantisation_backend(
                FPFormat(4, 3),
                FPFormat(5, 2),
                # FPFormat(2, 1), FPFormat(2, 1),
            )(gm, example_inputs)
        nonlocal graph_module
        graph_module = gm
        return gm

    torch._dynamo.reset()
    compiled = torch._dynamo.optimize(backend=_backend)(model)
    if unit_scaled:
        compiled.forward = unit_scaling.transforms.utils.patch_to_expand_modules(
            compiled.forward,
            non_recurse_functions=unit_scaling.transforms.utils._unit_scaled_functions,
        )

    # Trigger compilation with CPU tensors (of appropriate compute batch size)
    compiled(*(t[:COMPUTE_BATCH_SIZE] for t in next(iter(batches("val")))))

    return graph_module


def lr_schedule_fn(step: int) -> float:
    if step < WARMUP_ITERS:
        return step / WARMUP_ITERS
    return MIN_LR / LR + (1 - MIN_LR / LR) * (
        0.5 + 0.5 * np.cos(np.pi * (step - WARMUP_ITERS) / (MAX_ITERS - WARMUP_ITERS))
    )


def train(model: nanoGPT.model.GPT) -> Iterable[float]:
    opt = poptorch.optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.99))
    lr_schedule = torch.optim.lr_scheduler.LambdaLR(opt, lr_schedule_fn)
    options = poptorch.Options()
    options.replicationFactor(REPLICATION_FACTOR)
    options.outputMode(poptorch.OutputMode.All)
    training_options, inference_options = options.clone(), options.clone()
    steps = BATCH_SIZE // (COMPUTE_BATCH_SIZE * options.replication_factor)
    training_options.Training.gradientAccumulation(steps)
    inference_options.deviceIterations(steps)
    trainer = poptorch.trainingModel(model, options=training_options, optimizer=opt)
    evaluator = poptorch.inferenceModel(model, options=inference_options)

    def _losses() -> Iterable[float]:
        t0 = time.time()
        for n, batch in enumerate(batches("train")):
            out = dict(iter=n, lr=lr_schedule.get_last_lr()[0])
            if n % EVAL_INTERVAL == 0 and EVAL_ITERS:
                if n:
                    trainer.detachFromDevice()
                for split in ["train", "val"]:
                    out[f"{split}/loss"] = float(
                        torch.mean(
                            torch.stack(
                                [
                                    evaluator(*b)[1]
                                    for b in islice(batches(split), EVAL_ITERS)
                                ]
                            )
                        )
                    )
                evaluator.detachFromDevice()
            if n < MAX_ITERS:
                out.update(loss=float(torch.mean(trainer(*batch)[1])))
            yield dict(**out, t=time.time() - t0)
            if n >= MAX_ITERS:
                break
            lr_schedule.step()
            trainer.setOptimizer(opt)

    try:
        losses = iter(_losses())
        yield next(losses)
        yield from tqdm.tqdm(losses, initial=1, total=MAX_ITERS)
    finally:
        trainer.destroy()


if __name__ == "__main__":
    _model = nanoGPT.model.GPT(CONFIG)
    _model = transform(_model, unit_scaled=True, fake_fp8=True)
    for _line in train(_model):
        print(_line)
