Copyright (c) 2023 Graphcore Ltd. Licensed under the Apache 2.0 License.

The included code is released under an Apache 2.0 license, (see [LICENSE](LICENSE)).

Our dependencies are (see [pyproject.toml](pyproject.toml)):

| Component | About | License |
| --- | --- | --- |
| docstring-parser | Parse Python docstrings | MIT |
| einops | Deep learning operations reinvented (for pytorch, tensorflow, jax and others) | MIT |
| numpy | Array processing library | BSD 3-Clause |

We also use additional Python dependencies for development/testing (see [requirements-dev.txt](requirements-dev.txt)).

**This directory includes derived work from the following:**

---

Sphinx: https://github.com/sphinx-doc/sphinx, licensed under:

> Unless otherwise indicated, all code in the Sphinx project is licenced under the
> two clause BSD licence below.
> 
> Copyright (c) 2007-2023 by the Sphinx team (see AUTHORS file).
> All rights reserved.
> 
> Redistribution and use in source and binary forms, with or without
> modification, are permitted provided that the following conditions are
> met:
> 
> * Redistributions of source code must retain the above copyright
>   notice, this list of conditions and the following disclaimer.
> 
> * Redistributions in binary form must reproduce the above copyright
>   notice, this list of conditions and the following disclaimer in the
>   documentation and/or other materials provided with the distribution.
> 
> THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
> "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
> LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
> A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
> HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
> SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
> LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
> DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
> THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
> (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
> OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

this applies to:
- `docs/_templates/custom-class-template.rst` (modified)
- `docs/_templates/custom-module-template.rst` (modified)

---

The Example: Basic Sphinx project for Read the Docs: https://github.com/readthedocs-examples/example-sphinx-basic, licensed under:

> MIT License
> 
> Copyright (c) 2022 Read the Docs Inc
> 
> Permission is hereby granted, free of charge, to any person obtaining a copy
> of this software and associated documentation files (the "Software"), to deal
> in the Software without restriction, including without limitation the rights
> to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
> copies of the Software, and to permit persons to whom the Software is
> furnished to do so, subject to the following conditions:
> 
> The above copyright notice and this permission notice shall be included in all
> copies or substantial portions of the Software.
> 
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
> IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
> FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
> AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
> LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
> OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
> SOFTWARE.

this applies to:
- `docs/conf.py` (modified)
- `docs/make.bat`
- `docs/Makefile`
