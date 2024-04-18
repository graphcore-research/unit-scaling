# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

import random

import numpy as np
import pytest
import torch


@pytest.fixture(scope="function", autouse=True)
def fix_seed() -> None:
    """For each test function, reset all random seeds."""
    random.seed(1472)
    np.random.seed(1472)
    torch.manual_seed(1472)
