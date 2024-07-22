# Copyright (c) 2024 Graphcore Ltd. All rights reserved.

import copy
import pickle

import torch

from ..parameter import Parameter, has_parameter_data


def test_parameter() -> None:
    param = Parameter(torch.zeros(10), "weight")
    assert has_parameter_data(param)
    assert param.mup_type == "weight"
    assert param.mup_scaling_depth is None

    param.mup_scaling_depth = 8

    param_copy = copy.deepcopy(param)
    assert has_parameter_data(param_copy)  # type:ignore[arg-type]
    assert param_copy.mup_type == "weight"
    assert param_copy.mup_scaling_depth == 8

    param_pickle = pickle.loads(pickle.dumps(param))
    assert has_parameter_data(param_pickle)
    assert param_pickle.mup_type == "weight"
    assert param_pickle.mup_scaling_depth == 8

    param_pickle_copy = copy.deepcopy(param_pickle)
    assert has_parameter_data(param_pickle_copy)  # type:ignore[arg-type]
    assert param_pickle_copy.mup_type == "weight"
    assert param_pickle_copy.mup_scaling_depth == 8
