# Development

For users who wish to develop using this codebase, the following setup is required:

**First-time setup**:

```bash
python3 -m venv .venv
echo "export PYTHONPATH=\${PYTHONPATH}:\$(dirname \${VIRTUAL_ENV})" >> .venv/bin/activate
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