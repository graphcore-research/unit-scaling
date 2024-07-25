# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

from pathlib import Path
import re
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

version = re.search("__version__ = \"(.+)\"", Path("unit_scaling/_version.py").read_text()).group(1)

setuptools.setup(
    name="unit-scaling",
    version=version,
    description="A library for unit scaling in PyTorch.",
    packages=["unit_scaling", "unit_scaling.core", "unit_scaling.transforms"],
    install_requires=requirements,
)
