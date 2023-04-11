# Unit Scaling

A library for unit scaling in PyTorch.

Based on the paper
[Unit Scaling: Out-of-the-Box Low-Precision Training](https://arxiv.org/abs/2303.11257).

## Development

**First-time setup**:

```bash
python3 -m venv .venv
# Add to `.venv/bin/activate`: `source /PATH_TO_POPLAR_SDK/enable` (If running on IPU) 
source .venv/bin/activate

# pip install wheel  # (If running on IPU)
# pip install $POPLAR_SDK_ENABLED/../poptorch-*.whl  # (If running on IPU) 
pip install -r requirements-dev.txt
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
 - Consider a `.env` file for setting PYTHONPATH, e.g. `echo "PYTHONPATH=$(pwd)" > .env`
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