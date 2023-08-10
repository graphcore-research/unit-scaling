# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

from pathlib import Path

import setuptools

requirements = Path("requirements.txt").read_text().rstrip("\n").split("\n")
try:
    import poptorch

    # This should match requirements-dev-ipu.txt
    requirements.append(
        "poptorch-experimental-addons @ git+https://github.com/graphcore-research/poptorch-experimental-addons@beb12678d1e7ea2c033bd061d32167be262dfa58"
    )
except ImportError:
    pass

setuptools.setup(
    name="unit-scaling",
    description="A library for unit scaling in PyTorch.",
    packages=["unit_scaling", "unit_scaling.transforms"],
    install_requires=requirements,
)
