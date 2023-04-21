Search.setIndex({"docnames": ["generated/unit_scaling", "generated/unit_scaling.constraints", "generated/unit_scaling.constraints.amean", "generated/unit_scaling.constraints.gmean", "generated/unit_scaling.constraints.hmean", "generated/unit_scaling.constraints.to_grad_input_scale", "generated/unit_scaling.constraints.to_output_scale", "generated/unit_scaling.functional", "generated/unit_scaling.functional.gelu", "generated/unit_scaling.functional.linear", "generated/unit_scaling.functional.scale_elementwise", "generated/unit_scaling.modules", "generated/unit_scaling.modules.GELU", "generated/unit_scaling.modules.Linear", "generated/unit_scaling.modules.MLP", "generated/unit_scaling.scale", "generated/unit_scaling.scale.scale_bwd", "generated/unit_scaling.scale.scale_fwd", "generated/unit_scaling.utils", "generated/unit_scaling.utils.ScalePair", "generated/unit_scaling.utils.ScaleTracker", "generated/unit_scaling.utils.ScaleTrackingInterpreter", "generated/unit_scaling.utils.analyse_module", "index"], "filenames": ["generated/unit_scaling.rst", "generated/unit_scaling.constraints.rst", "generated/unit_scaling.constraints.amean.rst", "generated/unit_scaling.constraints.gmean.rst", "generated/unit_scaling.constraints.hmean.rst", "generated/unit_scaling.constraints.to_grad_input_scale.rst", "generated/unit_scaling.constraints.to_output_scale.rst", "generated/unit_scaling.functional.rst", "generated/unit_scaling.functional.gelu.rst", "generated/unit_scaling.functional.linear.rst", "generated/unit_scaling.functional.scale_elementwise.rst", "generated/unit_scaling.modules.rst", "generated/unit_scaling.modules.GELU.rst", "generated/unit_scaling.modules.Linear.rst", "generated/unit_scaling.modules.MLP.rst", "generated/unit_scaling.scale.rst", "generated/unit_scaling.scale.scale_bwd.rst", "generated/unit_scaling.scale.scale_fwd.rst", "generated/unit_scaling.utils.rst", "generated/unit_scaling.utils.ScalePair.rst", "generated/unit_scaling.utils.ScaleTracker.rst", "generated/unit_scaling.utils.ScaleTrackingInterpreter.rst", "generated/unit_scaling.utils.analyse_module.rst", "index.rst"], "titles": ["unit_scaling", "unit_scaling.constraints", "unit_scaling.constraints.amean", "unit_scaling.constraints.gmean", "unit_scaling.constraints.hmean", "unit_scaling.constraints.to_grad_input_scale", "unit_scaling.constraints.to_output_scale", "unit_scaling.functional", "unit_scaling.functional.gelu", "unit_scaling.functional.linear", "unit_scaling.functional.scale_elementwise", "unit_scaling.modules", "unit_scaling.modules.GELU", "unit_scaling.modules.Linear", "unit_scaling.modules.MLP", "unit_scaling.scale", "unit_scaling.scale.scale_bwd", "unit_scaling.scale.scale_fwd", "unit_scaling.utils", "unit_scaling.utils.ScalePair", "unit_scaling.utils.ScaleTracker", "unit_scaling.utils.ScaleTrackingInterpreter", "unit_scaling.utils.analyse_module", "Unit Scaling"], "terms": {"modul": [0, 21, 22], "common": [1, 7, 11], "scale": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 18, 22], "us": [1, 4, 12, 13, 20, 21, 22, 23], "unit": [1, 3, 7, 8, 9, 11, 12, 13, 14, 18], "oper": [1, 9, 15, 20, 22], "function": [1, 12, 13, 14, 15, 18, 20, 21, 22], "float": [2, 3, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 19, 21], "sourc": [2, 3, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 19, 20, 21, 22, 23], "comput": [2, 3, 4, 20], "arithmet": 2, "mean": [2, 3, 4, 9, 12, 13], "provid": [2, 3, 4, 5, 6, 8, 9, 10, 12, 13, 14], "paramet": [2, 3, 4, 5, 6, 9, 10, 12, 13, 14, 16, 17, 21, 22], "group": [2, 3, 4], "constrain": [2, 3, 4, 8, 9, 10, 12, 13, 14], "return": [2, 3, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 20, 21, 22], "type": [2, 3, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 21, 22], "geometr": 3, "recommend": [3, 23], "harmon": 4, "xavier": 4, "glorot": 4, "output_scal": [5, 6, 8, 9, 10, 12, 13, 14], "grad_input_scal": [5, 6, 8, 9, 10, 12, 13, 14], "assum": [5, 6], "two": 5, "select": [5, 6], "onli": [5, 6, 16, 17, 20, 21], "chosen": [5, 6], "factor": [5, 6, 14, 15, 16, 17], "op": [5, 6, 20, 21], "": [5, 6, 9, 14, 20, 21, 22], "output": [5, 6, 9, 10, 12, 13, 20, 21, 22], "input": [5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 20, 21, 22], "gradient": [5, 6, 8, 9, 10, 12, 13, 14, 20], "equal": [5, 6], "an": [6, 10, 13, 14, 19, 20, 21], "i": [6, 8, 9, 10, 12, 13, 14, 20, 21, 23], "ani": [6, 9, 12, 13, 20, 21], "number": [6, 9, 12, 13], "grad": [6, 10, 16, 17, 20], "version": [7, 10, 11], "torch": [7, 8, 9, 10, 11, 12, 13, 14, 20, 22], "nn": [7, 11, 12, 13, 14, 20, 21, 22], "tensor": [8, 9, 10, 13, 16, 17, 19, 20, 21, 22], "constraint": [8, 9, 10, 12, 13, 14], "option": [8, 9, 10, 12, 13, 14, 19, 21, 22], "callabl": [8, 9, 10, 12, 13, 14, 21], "gmean": [8, 9, 10, 12, 13, 14], "appli": [8, 9, 10, 12, 13, 14, 16, 17, 20], "when": [8, 12, 13, 21], "approxim": [8, 12, 14], "argument": [8, 12, 20, 21], "none": [8, 9, 10, 12, 13, 14, 19, 20, 21], "element": [8, 10], "wise": [8, 10], "text": [8, 12, 13], "x": [8, 12, 20, 22], "phi": [8, 12], "where": [8, 9, 12, 13], "cumul": [8, 12], "distribut": [8, 12], "gaussian": [8, 12], "tanh": [8, 12], "estim": [8, 12], "math": 8, "0": [8, 12, 20, 22], "5": [8, 12], "1": [8, 12, 13, 20, 22], "sqrt": [8, 12, 13], "2": [8, 9, 12, 20, 22], "pi": [8, 12], "044715": [8, 12], "3": [8, 12], "see": [8, 20, 21, 23], "error": [8, 12, 20, 21], "linear": [8, 12, 22], "param": 8, "whichtak": [8, 9, 10, 12, 13, 14], "singl": [8, 9, 10, 12, 13, 14], "usual": [8, 9, 10, 12, 13, 14], "necessari": [8, 9, 10, 12, 13, 14, 20], "valid": [8, 9, 10, 12, 13, 14], "If": [8, 9, 10, 12, 13, 14, 20, 21, 23], "default": [8, 9, 10, 12, 13, 14, 20, 22], "binaryconstraint": [8, 9, 10, 12, 13, 14], "weight": [9, 13, 22], "bia": [9, 13, 22], "transform": [9, 10, 13], "incom": [9, 13], "data": [9, 13, 21], "y": [9, 20], "xa": 9, "t": [9, 20], "b": [9, 20], "thi": [9, 13, 20, 21, 23], "opper": 9, "support": [9, 13, 20, 23], "d": [9, 20, 22], "spars": 9, "layout": [9, 21], "beta": 9, "featur": 9, "some": 9, "dtype": [9, 13, 20, 21], "devic": [9, 13, 21], "combin": 9, "mai": [9, 20], "have": [9, 20], "autograd": [9, 20], "you": [9, 20], "notic": [9, 23], "miss": 9, "pleas": 9, "open": 9, "request": 9, "tensorfloat32": [9, 13], "shape": [9, 12, 13, 20], "_featur": [9, 13], "addit": [9, 13], "dimens": [9, 12, 13, 14], "includ": [9, 13], "out": [9, 13, 20, 23], "base": [9, 20, 23], "f": 10, "should": [10, 20, 21], "take": 10, "its": [10, 17, 20], "first": [10, 20, 21, 23], "follow": [10, 20], "arg": [10, 20, 21], "kwarg": [10, 20, 21], "class": [11, 12, 13, 14, 18, 19, 20, 21, 22], "str": [12, 21, 22], "algorithm": 12, "same": [12, 13, 20], "exampl": [12, 13, 20, 21, 22], "m": [12, 13, 23], "randn": [12, 13, 22], "in_featur": 13, "int": [13, 14, 20, 21], "out_featur": 13, "bool": [13, 20, 21, 22], "true": [13, 20, 21, 22], "On": 13, "certain": [13, 21], "rocm": 13, "float16": 13, "differ": [13, 15, 23], "precis": [13, 23], "backward": [13, 15, 16, 19, 20, 21, 22], "size": [13, 14], "each": [13, 20, 21, 22], "sampl": 13, "set": [13, 20, 23], "fals": [13, 20, 22], "layer": [13, 14], "learn": 13, "learnabl": 13, "The": [13, 20, 21], "valu": [13, 20, 21], "ar": [13, 20], "initi": 13, "from": [13, 20, 21], "mathcal": 13, "u": 13, "k": 13, "frac": 13, "h_": 13, "all": [13, 20], "last": 13, "20": 13, "30": 13, "128": 13, "print": [13, 22], "hidden_s": [14, 22], "act_fn": 14, "gelu": 14, "expansion_factor": 14, "4": [14, 20, 22], "A": [14, 23], "implement": [14, 22], "hidden": 14, "activ": [14, 23], "which": [14, 21], "intermedi": [14, 21], "increas": [14, 20], "rel": 14, "enabl": [15, 20, 23], "forward": [15, 16, 17, 19, 20, 21, 22], "pass": [15, 16, 17, 19, 20, 21, 22], "scalar": [16, 17, 19], "multipl": [16, 17], "unchang": 16, "origin": 17, "develop": 18, "model": 18, "dataclass": 19, "contain": 19, "pair": 19, "intend": [19, 20], "repres": [19, 20, 22], "standard": [19, 20, 21, 22], "deviat": [19, 20, 21, 22], "arbitrari": 19, "given": [20, 22], "record": [20, 21], "suppli": 20, "dict": [20, 21], "static": 20, "ctx": 20, "functionctx": 20, "tupl": [20, 21], "defin": 20, "formula": 20, "differenti": 20, "mode": 20, "automat": 20, "alia": 20, "vjp": 20, "overridden": 20, "subclass": 20, "It": 20, "must": 20, "accept": 20, "context": 20, "mani": 20, "non": 20, "were": 20, "w": 20, "r": [20, 23], "correspond": 20, "requir": [20, 23], "can": [20, 21], "just": [20, 21], "retriev": [20, 21], "save": [20, 23], "dure": 20, "also": 20, "ha": 20, "attribut": [20, 21], "needs_input_grad": 20, "boolean": 20, "whether": 20, "need": 20, "e": [20, 23], "g": [20, 23], "jvp": 20, "grad_input": 20, "got": 20, "respect": 20, "object": 20, "mark_dirti": 20, "mark": 20, "modifi": 20, "place": 20, "call": [20, 21], "most": 20, "onc": 20, "insid": 20, "method": [20, 21, 22], "everi": [20, 21], "been": 20, "ensur": 20, "correct": 20, "our": 20, "check": [20, 23], "doesn": 20, "matter": 20, "befor": [20, 21], "after": 20, "modif": 20, "inplac": [20, 22], "staticmethod": 20, "def": [20, 22], "x_npy": 20, "numpi": 20, "share": 20, "storag": 20, "once_differenti": 20, "grad_output": 20, "requires_grad": 20, "doubl": 20, "clone": 20, "would": 20, "lead": 20, "wrong": 20, "engin": 20, "know": 20, "unless": 20, "we": [20, 21], "xdoctest": 20, "skip": 20, "runtimeerror": 20, "one": 20, "variabl": 20, "mark_non_differenti": 20, "effici": 20, "still": 20, "alwai": 20, "go": 20, "zero": 20, "indic": 20, "sort": 20, "func": 20, "idx": 20, "save_for_backward": 20, "g1": 20, "g2": 20, "saved_tensor": 20, "zeros_lik": 20, "index_add_": 20, "futur": 20, "oppos": 20, "directli": 20, "prevent": 20, "incorrect": 20, "memori": 20, "leak": 20, "applic": 20, "hook": 20, "graph": [20, 21], "saved_tensors_hook": 20, "note": [20, 21, 23], "intermediari": 20, "neither": 20, "nor": 20, "your": [20, 23], "custom": 20, "do": [20, 21], "decor": 20, "so": [20, 21], "perform": 20, "rais": 20, "like": 20, "either": 20, "recomput": 20, "tutori": 20, "more": 20, "detail": [20, 21, 23], "In": 20, "access": 20, "through": [20, 21], "them": [20, 21], "user": 20, "made": 20, "thei": 20, "weren": 20, "content": 20, "extend": 20, "how": 20, "z": 20, "grad_out": 20, "gx": 20, "gy": 20, "gz": 20, "c": [20, 23], "save_for_forward": 20, "x_t": 20, "y_t": 20, "_": 20, "fwad": 20, "dual_level": 20, "a_dual": 20, "make_du": 20, "set_materialize_grad": 20, "materi": 20, "undefin": 20, "expand": 20, "full": 20, "prior": 20, "simplefunc": 20, "No": 20, "handl": 20, "now": 20, "induc": 20, "graphmodul": 21, "wrap": 21, "fx": [21, 22], "than": 21, "execut": 21, "instrument": 21, "call_funct": 21, "target": 21, "union": 21, "list": 21, "slice": 21, "node": 21, "complex": 21, "memory_format": 21, "result": 21, "semant": 21, "posit": 21, "invoc": 21, "keyword": 21, "compat": 21, "api": 21, "guarante": 21, "call_method": 21, "call_modul": 21, "fetch_args_kwargs_from_env": 21, "n": 21, "fetch": 21, "concret": 21, "current": 21, "environ": 21, "fetch_attr": 21, "hierarchi": 21, "self": [21, 22], "fulli": 21, "qualfii": 21, "name": 21, "get_attr": 21, "Will": 21, "wa": 21, "map_nodes_to_valu": 21, "recurs": [21, 22], "descend": 21, "look": 21, "up": 21, "structur": 21, "within": 21, "belong": 21, "report": 21, "realli": 21, "referenc": 21, "placehold": 21, "state": 21, "interpret": 21, "maintain": 21, "intern": 21, "iter": 21, "over": 21, "run": [21, 23], "next": 21, "initial_env": 21, "enable_io_process": 21, "via": 21, "order": 21, "start": 21, "map": 21, "pre": [21, 23], "popul": 21, "partial": 21, "evalu": 21, "process": 21, "process_input": 21, "process_output": 21, "run_nod": 21, "specif": 21, "depend": 21, "recurse_modul": 22, "syntax_highlight": 22, "dummi": 22, "gener": 22, "code": 22, "annot": 22, "both": 22, "analys": 22, "fed": 22, "analysi": 22, "toggl": 22, "behavour": 22, "string": 22, "reflect": 22, "mlp": 22, "__init__": 22, "super": 22, "fc1": 22, "relu": 22, "fc2": 22, "10": 22, "requires_grad_": 22, "bwd": 22, "236": 22, "fc1_weight": 22, "018": 22, "6": 22, "54": 22, "fc1_bia": 22, "0182": 22, "51": 22, "_c": 22, "_nn": 22, "578": 22, "204": 22, "337": 22, "288": 22, "fc2_weight": 22, "00902": 22, "13": 22, "fc2_bia": 22, "00904": 22, "31": 22, "linear_1": 22, "235": 22, "999": 22, "librari": 23, "pytorch": 23, "paper": 23, "box": 23, "low": 23, "train": 23, "time": 23, "setup": 23, "python3": 23, "venv": 23, "add": 23, "bin": 23, "path_to_poplar_sdk": 23, "ipu": 23, "pip": 23, "instal": 23, "wheel": 23, "poplar_sdk_en": 23, "poptorch": 23, "whl": 23, "dev": 23, "txt": 23, "subsequ": 23, "flight": 23, "help": 23, "command": 23, "id": 23, "python": 23, "intepret": 23, "format": 23, "consid": 23, "env": 23, "file": 23, "pythonpath": 23, "echo": 23, "pwd": 23, "path": 23, "devcontain": 23, "doc": 23, "cd": 23, "make": 23, "html": 23, "view": 23, "_build": 23, "index": 23, "browser": 23, "copyright": 23, "2023": 23, "graphcor": 23, "ltd": 23, "under": 23, "mit": 23, "md": 23, "further": 23}, "objects": {"": [[0, 0, 0, "-", "unit_scaling"]], "unit_scaling": [[1, 0, 0, "-", "constraints"], [7, 0, 0, "-", "functional"], [11, 0, 0, "-", "modules"], [15, 0, 0, "-", "scale"], [18, 0, 0, "-", "utils"]], "unit_scaling.constraints": [[2, 1, 1, "", "amean"], [3, 1, 1, "", "gmean"], [4, 1, 1, "", "hmean"], [5, 1, 1, "", "to_grad_input_scale"], [6, 1, 1, "", "to_output_scale"]], "unit_scaling.functional": [[8, 1, 1, "", "gelu"], [9, 1, 1, "", "linear"], [10, 1, 1, "", "scale_elementwise"]], "unit_scaling.modules": [[12, 2, 1, "", "GELU"], [13, 2, 1, "", "Linear"], [14, 2, 1, "", "MLP"]], "unit_scaling.modules.Linear": [[13, 3, 1, "", "bias"], [13, 3, 1, "", "weight"]], "unit_scaling.scale": [[16, 1, 1, "", "scale_bwd"], [17, 1, 1, "", "scale_fwd"]], "unit_scaling.utils": [[19, 2, 1, "", "ScalePair"], [20, 2, 1, "", "ScaleTracker"], [21, 2, 1, "", "ScaleTrackingInterpreter"], [22, 1, 1, "", "analyse_module"]], "unit_scaling.utils.ScaleTracker": [[20, 4, 1, "", "backward"], [20, 4, 1, "", "jvp"], [20, 4, 1, "", "mark_dirty"], [20, 4, 1, "", "mark_non_differentiable"], [20, 4, 1, "", "save_for_backward"], [20, 4, 1, "", "save_for_forward"], [20, 4, 1, "", "set_materialize_grads"], [20, 4, 1, "", "vjp"]], "unit_scaling.utils.ScaleTrackingInterpreter": [[21, 4, 1, "", "call_function"], [21, 4, 1, "", "call_method"], [21, 4, 1, "", "call_module"], [21, 4, 1, "", "fetch_args_kwargs_from_env"], [21, 4, 1, "", "fetch_attr"], [21, 4, 1, "", "get_attr"], [21, 4, 1, "", "map_nodes_to_values"], [21, 4, 1, "", "output"], [21, 4, 1, "", "placeholder"], [21, 4, 1, "", "run"], [21, 4, 1, "", "run_node"]]}, "objtypes": {"0": "py:module", "1": "py:function", "2": "py:class", "3": "py:attribute", "4": "py:method"}, "objnames": {"0": ["py", "module", "Python module"], "1": ["py", "function", "Python function"], "2": ["py", "class", "Python class"], "3": ["py", "attribute", "Python attribute"], "4": ["py", "method", "Python method"]}, "titleterms": {"unit_sc": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22], "constraint": [1, 2, 3, 4, 5, 6], "amean": 2, "gmean": 3, "hmean": 4, "to_grad_input_scal": 5, "to_output_scal": 6, "function": [7, 8, 9, 10], "gelu": [8, 12], "linear": [9, 13], "scale_elementwis": 10, "modul": [11, 12, 13, 14], "mlp": 14, "scale": [15, 16, 17, 23], "scale_bwd": 16, "scale_fwd": 17, "util": [18, 19, 20, 21, 22], "scalepair": 19, "scaletrack": 20, "scaletrackinginterpret": 21, "analyse_modul": 22, "unit": 23, "develop": 23, "licens": 23, "api": 23}, "envversion": {"sphinx.domains.c": 2, "sphinx.domains.changeset": 1, "sphinx.domains.citation": 1, "sphinx.domains.cpp": 8, "sphinx.domains.index": 1, "sphinx.domains.javascript": 2, "sphinx.domains.math": 2, "sphinx.domains.python": 3, "sphinx.domains.rst": 2, "sphinx.domains.std": 2, "sphinx.ext.intersphinx": 1, "sphinx.ext.viewcode": 1, "sphinx": 57}, "alltitles": {"unit_scaling": [[0, "module-unit_scaling"]], "unit_scaling.constraints": [[1, "module-unit_scaling.constraints"]], "unit_scaling.constraints.amean": [[2, "unit-scaling-constraints-amean"]], "unit_scaling.constraints.gmean": [[3, "unit-scaling-constraints-gmean"]], "unit_scaling.constraints.hmean": [[4, "unit-scaling-constraints-hmean"]], "unit_scaling.constraints.to_grad_input_scale": [[5, "unit-scaling-constraints-to-grad-input-scale"]], "unit_scaling.constraints.to_output_scale": [[6, "unit-scaling-constraints-to-output-scale"]], "unit_scaling.functional": [[7, "module-unit_scaling.functional"]], "unit_scaling.functional.gelu": [[8, "unit-scaling-functional-gelu"]], "unit_scaling.functional.linear": [[9, "unit-scaling-functional-linear"]], "unit_scaling.functional.scale_elementwise": [[10, "unit-scaling-functional-scale-elementwise"]], "unit_scaling.modules": [[11, "module-unit_scaling.modules"]], "unit_scaling.modules.GELU": [[12, "unit-scaling-modules-gelu"]], "unit_scaling.modules.Linear": [[13, "unit-scaling-modules-linear"]], "unit_scaling.modules.MLP": [[14, "unit-scaling-modules-mlp"]], "unit_scaling.scale": [[15, "module-unit_scaling.scale"]], "unit_scaling.scale.scale_bwd": [[16, "unit-scaling-scale-scale-bwd"]], "unit_scaling.scale.scale_fwd": [[17, "unit-scaling-scale-scale-fwd"]], "unit_scaling.utils": [[18, "module-unit_scaling.utils"]], "unit_scaling.utils.ScalePair": [[19, "unit-scaling-utils-scalepair"]], "unit_scaling.utils.ScaleTracker": [[20, "unit-scaling-utils-scaletracker"]], "unit_scaling.utils.ScaleTrackingInterpreter": [[21, "unit-scaling-utils-scaletrackinginterpreter"]], "unit_scaling.utils.analyse_module": [[22, "unit-scaling-utils-analyse-module"]], "Unit Scaling": [[23, "unit-scaling"]], "Development": [[23, "development"]], "License": [[23, "license"]], "API": [[23, "api"]]}, "indexentries": {"module": [[0, "module-unit_scaling"], [1, "module-unit_scaling.constraints"], [7, "module-unit_scaling.functional"], [11, "module-unit_scaling.modules"], [15, "module-unit_scaling.scale"], [18, "module-unit_scaling.utils"]], "unit_scaling": [[0, "module-unit_scaling"]], "unit_scaling.constraints": [[1, "module-unit_scaling.constraints"]], "amean() (in module unit_scaling.constraints)": [[2, "unit_scaling.constraints.amean"]], "gmean() (in module unit_scaling.constraints)": [[3, "unit_scaling.constraints.gmean"]], "hmean() (in module unit_scaling.constraints)": [[4, "unit_scaling.constraints.hmean"]], "to_grad_input_scale() (in module unit_scaling.constraints)": [[5, "unit_scaling.constraints.to_grad_input_scale"]], "to_output_scale() (in module unit_scaling.constraints)": [[6, "unit_scaling.constraints.to_output_scale"]], "unit_scaling.functional": [[7, "module-unit_scaling.functional"]], "gelu() (in module unit_scaling.functional)": [[8, "unit_scaling.functional.gelu"]], "linear() (in module unit_scaling.functional)": [[9, "unit_scaling.functional.linear"]], "scale_elementwise() (in module unit_scaling.functional)": [[10, "unit_scaling.functional.scale_elementwise"]], "unit_scaling.modules": [[11, "module-unit_scaling.modules"]], "gelu (class in unit_scaling.modules)": [[12, "unit_scaling.modules.GELU"]], "linear (class in unit_scaling.modules)": [[13, "unit_scaling.modules.Linear"]], "bias (unit_scaling.modules.linear attribute)": [[13, "unit_scaling.modules.Linear.bias"]], "weight (unit_scaling.modules.linear attribute)": [[13, "unit_scaling.modules.Linear.weight"]], "mlp (class in unit_scaling.modules)": [[14, "unit_scaling.modules.MLP"]], "unit_scaling.scale": [[15, "module-unit_scaling.scale"]], "scale_bwd() (in module unit_scaling.scale)": [[16, "unit_scaling.scale.scale_bwd"]], "scale_fwd() (in module unit_scaling.scale)": [[17, "unit_scaling.scale.scale_fwd"]], "unit_scaling.utils": [[18, "module-unit_scaling.utils"]], "scalepair (class in unit_scaling.utils)": [[19, "unit_scaling.utils.ScalePair"]], "scaletracker (class in unit_scaling.utils)": [[20, "unit_scaling.utils.ScaleTracker"]], "backward() (unit_scaling.utils.scaletracker static method)": [[20, "unit_scaling.utils.ScaleTracker.backward"]], "jvp() (unit_scaling.utils.scaletracker static method)": [[20, "unit_scaling.utils.ScaleTracker.jvp"]], "mark_dirty() (unit_scaling.utils.scaletracker method)": [[20, "unit_scaling.utils.ScaleTracker.mark_dirty"]], "mark_non_differentiable() (unit_scaling.utils.scaletracker method)": [[20, "unit_scaling.utils.ScaleTracker.mark_non_differentiable"]], "save_for_backward() (unit_scaling.utils.scaletracker method)": [[20, "unit_scaling.utils.ScaleTracker.save_for_backward"]], "save_for_forward() (unit_scaling.utils.scaletracker method)": [[20, "unit_scaling.utils.ScaleTracker.save_for_forward"]], "set_materialize_grads() (unit_scaling.utils.scaletracker method)": [[20, "unit_scaling.utils.ScaleTracker.set_materialize_grads"]], "vjp() (unit_scaling.utils.scaletracker static method)": [[20, "unit_scaling.utils.ScaleTracker.vjp"]], "scaletrackinginterpreter (class in unit_scaling.utils)": [[21, "unit_scaling.utils.ScaleTrackingInterpreter"]], "call_function() (unit_scaling.utils.scaletrackinginterpreter method)": [[21, "unit_scaling.utils.ScaleTrackingInterpreter.call_function"]], "call_method() (unit_scaling.utils.scaletrackinginterpreter method)": [[21, "unit_scaling.utils.ScaleTrackingInterpreter.call_method"]], "call_module() (unit_scaling.utils.scaletrackinginterpreter method)": [[21, "unit_scaling.utils.ScaleTrackingInterpreter.call_module"]], "fetch_args_kwargs_from_env() (unit_scaling.utils.scaletrackinginterpreter method)": [[21, "unit_scaling.utils.ScaleTrackingInterpreter.fetch_args_kwargs_from_env"]], "fetch_attr() (unit_scaling.utils.scaletrackinginterpreter method)": [[21, "unit_scaling.utils.ScaleTrackingInterpreter.fetch_attr"]], "get_attr() (unit_scaling.utils.scaletrackinginterpreter method)": [[21, "unit_scaling.utils.ScaleTrackingInterpreter.get_attr"]], "map_nodes_to_values() (unit_scaling.utils.scaletrackinginterpreter method)": [[21, "unit_scaling.utils.ScaleTrackingInterpreter.map_nodes_to_values"]], "output() (unit_scaling.utils.scaletrackinginterpreter method)": [[21, "unit_scaling.utils.ScaleTrackingInterpreter.output"]], "placeholder() (unit_scaling.utils.scaletrackinginterpreter method)": [[21, "unit_scaling.utils.ScaleTrackingInterpreter.placeholder"]], "run() (unit_scaling.utils.scaletrackinginterpreter method)": [[21, "unit_scaling.utils.ScaleTrackingInterpreter.run"]], "run_node() (unit_scaling.utils.scaletrackinginterpreter method)": [[21, "unit_scaling.utils.ScaleTrackingInterpreter.run_node"]], "analyse_module() (in module unit_scaling.utils)": [[22, "unit_scaling.utils.analyse_module"]]}})