# Unit-Scaled Maximal Update Parameterization (u-μP)

[![tests](https://github.com/graphcore-research/unit-scaling/actions/workflows/ci.yaml/badge.svg)](https://github.com/graphcore-research/unit-scaling/actions/workflows/ci-public.yaml)
![PyPI version](https://img.shields.io/pypi/v/unit-scaling)
[![license](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/graphcore-research/unit-scaling/blob/main/LICENSE)
[![GitHub Repo stars](https://img.shields.io/github/stars/graphcore-research/unit-scaling)](https://github.com/graphcore-research/unit-scaling/stargazers)

A library for unit scaling in PyTorch, based on the paper [u-μP: The Unit-Scaled Maximal Update Parametrization](https://arxiv.org/abs/2407.17465) and previous work [Unit Scaling: Out-of-the-Box Low-Precision Training](https://arxiv.org/abs/2303.11257).

Documentation can be found at
[https://graphcore-research.github.io/unit-scaling](https://graphcore-research.github.io/unit-scaling) and an example notebook at [examples/demo.ipynb](examples/demo.ipynb).

**Note:** The library is currently in its _beta_ release.
Some features have yet to be implemented and occasional bugs may be present.
We're keen to help users with any problems they encounter.

## Installation

To install the `unit-scaling` library, run:

```sh
pip install unit-scaling
```
or for a local editable install (i.e. one which uses the files in this repo), run:

```sh
pip install -e .
```

## Development

For development in this repository, we recommend using the provided docker container.
This image can be built and entered interactively using:

```sh
docker build -t unit-scaling-dev:latest .
docker run -it --rm  --user developer:developer -v $(pwd):/home/developer/unit-scaling unit-scaling-dev:latest
# To use git within the container, add `-v ~/.ssh:/home/developer/.ssh:ro -v ~/.gitconfig:/home/developer/.gitconfig:ro`.
```

For vscode users, this repo also contains a `.devcontainer.json` file, which enables the container to be used as a full-featured IDE (see the [Dev Container docs](https://code.visualstudio.com/docs/devcontainers/containers) for details on how to use this feature).

Key development functionality is contained within the `./dev` script. This includes running unit tests, linting, formatting, documentation generation and more. Run `./dev --help` for the available options. Running `./dev` without arguments is equivalent to using the `--ci` option, which runs all of the available dev checks. This is the test used for GitHub CI.

We encourage pull requests from the community. Please reach out to us with any questions about contributing.

## What is u-μP?

u-μP inserts scaling factors into the model to make activations, gradients and weights unit-scaled (RMS ≈ 1) at initialisation, and into optimiser learning rates to keep updates stable as models are scaled in width and depth. This results in hyperparameter transfer from small to large models and easy support for low-precision training.

For a quick intro, see [examples/demo.ipynb](examples/demo.ipynb), for more depth see the [paper](https://arxiv.org/abs/2407.17465) and [library documentation](https://graphcore-research.github.io/unit-scaling/).

## What is unit scaling?

For a demonstration of the library and an overview of how it works, see
[Out-of-the-Box FP8 Training](https://github.com/graphcore-research/out-of-the-box-fp8-training/blob/main/out_of_the_box_fp8_training.ipynb)
(a notebook showing how to unit-scale the nanoGPT model).

For a more in-depth explanation, consult our paper
[Unit Scaling: Out-of-the-Box Low-Precision Training](https://arxiv.org/abs/2303.11257).

And for a practical introduction to using the library, see our [User Guide](https://graphcore-research.github.io/unit-scaling/user_guide.html).

## License

Copyright (c) 2023 Graphcore Ltd. Licensed under the Apache 2.0 License.

See [NOTICE.md](NOTICE.md) for further details.
