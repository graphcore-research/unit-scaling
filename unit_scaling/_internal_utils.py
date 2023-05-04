# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

import inspect
import sys
from typing import List


def generate__all__(module_name: str, include_imports: bool = False) -> List[str]:
    """Generates the contents of __all__ by extracting every public function/class/etc.
    except those imported from other modules. Necessary for Sphinx docs."""
    module = sys.modules[module_name]
    all = []
    for name, member in inspect.getmembers(module):
        # Skip members imported from other modules and private members
        is_local = inspect.getmodule(member) == module
        if (include_imports or is_local) and not name.startswith("_"):
            all.append(name)
    return all
