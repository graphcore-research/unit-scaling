# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

from pathlib import Path

import setuptools

setuptools.setup(
    name="unit-scaling",
    description="A library for unit scaling in PyTorch.",
    packages=["unit_scaling"],
    install_requires=Path("requirements.txt").read_text().rstrip("\n").split("\n"),
)
