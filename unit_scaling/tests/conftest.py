# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

import random

import numpy as np
import pytest
import torch

# Some tests depend on dynamo implementation details, so need to check PyTorch versions
pt20 = torch.__version__ >= "2.0" and torch.__version__ < "2.1"


@pytest.fixture(scope="function", autouse=True)
def fix_seed() -> None:
    """For each test function, reset all random seeds."""
    random.seed(1472)
    np.random.seed(1472)
    torch.manual_seed(1472)
