# Unit Scaling

A library for unit scaling in PyTorch, based on the paper
[Unit Scaling: Out-of-the-Box Low-Precision Training](https://arxiv.org/abs/2303.11257).

Documentation can be found at
[https://graphcore-research.github.io/unit-scaling](https://graphcore-research.github.io/unit-scaling).

**Note:** This library is currently in its *beta* release.
Some features have yet to be implemented and occasional bugs may be present.
We're keen to help users with any problems they encounter.

## Installation

To install the `unit-scaling` library, run:

```
pip install git+https://github.com/graphcore-research/unit-scaling.git
```

## What is unit scaling?

For a high-level introduction to unit scaling, see our blog post
[Simple FP16 and FP8 training with Unit Scaling](https://www.graphcore.ai/posts/simple-fp16-and-fp8-training-with-unit-scaling).

For a more in-depth explanation, consult our paper
[Unit Scaling: Out-of-the-Box Low-Precision Training](https://arxiv.org/abs/2303.11257).

And for a practical introduction to using the library, see our [User Guide](https://graphcore-research.github.io/unit-scaling/user_guide.html).

## Development

For users who wish to develop using this codebase, the following setup is required:

**First-time setup**:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt  # Or requirements-dev-ipu.txt for the ipu
```

**Subsequent setup**:

```bash
source .venv/bin/activate
```

**Run pre-flight checks** (or run `./dev --help` to see supported commands):

```bash
./dev
```

**IDE recommendations**:

- Python intepreter is set to `.venv/bin/python`
- Format-on-save enabled
- Consider a `.env` file for setting `PYTHONPATH`, for example `echo "PYTHONPATH=$(pwd)" > .env`
  (note that this will be a different path if using devcontainers)

**Docs development**:

```bash
cd docs/
make html
```

then view `docs/_build/html/index.html` in your browser.

## License

Copyright (c) 2023 Graphcore Ltd. Licensed under the MIT License.

See [NOTICE.md](NOTICE.md) for further details.
