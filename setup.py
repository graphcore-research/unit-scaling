# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

from pathlib import Path

import setuptools
import setuptools.command.build_ext

setuptools.setup(
    name="unit-scaling",
    description="A library for unit scaling in PyTorch.",
    version="0.1",
    packages=["unit-scaling"],
    install_requires=Path("requirements.txt").read_text().rstrip("\n").split("\n"),
)
