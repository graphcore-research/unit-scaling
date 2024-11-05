# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

from pathlib import Path
import re
import setuptools

requirements = Path("requirements.txt").read_text().rstrip("\n").split("\n")

version = re.search("__version__ = \"(.+)\"", Path("unit_scaling/_version.py").read_text()).group(1)

setuptools.setup(
    name="unit-scaling",
    version=version,
    description="A library for unit scaling in PyTorch.",
    packages=["unit_scaling", "unit_scaling.core", "unit_scaling.transforms"],
    install_requires=requirements,
)
