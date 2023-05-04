Search.setIndex({"docnames": ["generated/unit_scaling", "generated/unit_scaling.constraints", "generated/unit_scaling.constraints.amean", "generated/unit_scaling.constraints.gmean", "generated/unit_scaling.constraints.hmean", "generated/unit_scaling.constraints.to_grad_input_scale", "generated/unit_scaling.constraints.to_left_grad_scale", "generated/unit_scaling.constraints.to_output_scale", "generated/unit_scaling.constraints.to_right_grad_scale", "generated/unit_scaling.functional", "generated/unit_scaling.functional.cross_entropy", "generated/unit_scaling.functional.dropout", "generated/unit_scaling.functional.embedding", "generated/unit_scaling.functional.gelu", "generated/unit_scaling.functional.layer_norm", "generated/unit_scaling.functional.linear", "generated/unit_scaling.functional.matmul", "generated/unit_scaling.functional.residual_add", "generated/unit_scaling.functional.residual_split", "generated/unit_scaling.functional.scale_elementwise", "generated/unit_scaling.functional.softmax", "generated/unit_scaling.modules", "generated/unit_scaling.modules.MHSA", "generated/unit_scaling.modules.MLP", "generated/unit_scaling.modules.TransformerDecoder", "generated/unit_scaling.modules.TransformerLayer", "generated/unit_scaling.scale", "generated/unit_scaling.scale.scale_bwd", "generated/unit_scaling.scale.scale_fwd", "generated/unit_scaling.utils", "generated/unit_scaling.utils.ScalePair", "generated/unit_scaling.utils.ScaleTracker", "generated/unit_scaling.utils.ScaleTrackingInterpreter", "generated/unit_scaling.utils.analyse_module", "index"], "filenames": ["generated/unit_scaling.rst", "generated/unit_scaling.constraints.rst", "generated/unit_scaling.constraints.amean.rst", "generated/unit_scaling.constraints.gmean.rst", "generated/unit_scaling.constraints.hmean.rst", "generated/unit_scaling.constraints.to_grad_input_scale.rst", "generated/unit_scaling.constraints.to_left_grad_scale.rst", "generated/unit_scaling.constraints.to_output_scale.rst", "generated/unit_scaling.constraints.to_right_grad_scale.rst", "generated/unit_scaling.functional.rst", "generated/unit_scaling.functional.cross_entropy.rst", "generated/unit_scaling.functional.dropout.rst", "generated/unit_scaling.functional.embedding.rst", "generated/unit_scaling.functional.gelu.rst", "generated/unit_scaling.functional.layer_norm.rst", "generated/unit_scaling.functional.linear.rst", "generated/unit_scaling.functional.matmul.rst", "generated/unit_scaling.functional.residual_add.rst", "generated/unit_scaling.functional.residual_split.rst", "generated/unit_scaling.functional.scale_elementwise.rst", "generated/unit_scaling.functional.softmax.rst", "generated/unit_scaling.modules.rst", "generated/unit_scaling.modules.MHSA.rst", "generated/unit_scaling.modules.MLP.rst", "generated/unit_scaling.modules.TransformerDecoder.rst", "generated/unit_scaling.modules.TransformerLayer.rst", "generated/unit_scaling.scale.rst", "generated/unit_scaling.scale.scale_bwd.rst", "generated/unit_scaling.scale.scale_fwd.rst", "generated/unit_scaling.utils.rst", "generated/unit_scaling.utils.ScalePair.rst", "generated/unit_scaling.utils.ScaleTracker.rst", "generated/unit_scaling.utils.ScaleTrackingInterpreter.rst", "generated/unit_scaling.utils.analyse_module.rst", "index.rst"], "titles": ["unit_scaling", "unit_scaling.constraints", "unit_scaling.constraints.amean", "unit_scaling.constraints.gmean", "unit_scaling.constraints.hmean", "unit_scaling.constraints.to_grad_input_scale", "unit_scaling.constraints.to_left_grad_scale", "unit_scaling.constraints.to_output_scale", "unit_scaling.constraints.to_right_grad_scale", "unit_scaling.functional", "unit_scaling.functional.cross_entropy", "unit_scaling.functional.dropout", "unit_scaling.functional.embedding", "unit_scaling.functional.gelu", "unit_scaling.functional.layer_norm", "unit_scaling.functional.linear", "unit_scaling.functional.matmul", "unit_scaling.functional.residual_add", "unit_scaling.functional.residual_split", "unit_scaling.functional.scale_elementwise", "unit_scaling.functional.softmax", "unit_scaling.modules", "unit_scaling.modules.MHSA", "unit_scaling.modules.MLP", "unit_scaling.modules.TransformerDecoder", "unit_scaling.modules.TransformerLayer", "unit_scaling.scale", "unit_scaling.scale.scale_bwd", "unit_scaling.scale.scale_fwd", "unit_scaling.utils", "unit_scaling.utils.ScalePair", "unit_scaling.utils.ScaleTracker", "unit_scaling.utils.ScaleTrackingInterpreter", "unit_scaling.utils.analyse_module", "Unit Scaling"], "terms": {"modul": [0, 12, 16, 32, 33], "common": [1, 9, 21], "scale": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 23, 24, 25, 29, 33], "us": [1, 4, 11, 12, 16, 17, 18, 20, 22, 24, 25, 31, 32, 33, 34], "unit": [1, 3, 9, 10, 11, 12, 13, 14, 15, 16, 18, 20, 21, 22, 23, 24, 25, 29, 33], "oper": [1, 11, 15, 16, 20, 26, 31, 33], "function": [1, 22, 23, 24, 25, 26, 29, 31, 32, 33], "float": [2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 23, 24, 25, 27, 28, 30, 32], "sourc": [2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 23, 24, 25, 27, 28, 30, 31, 32, 33, 34], "comput": [2, 3, 4, 10, 12, 20, 31], "arithmet": 2, "mean": [2, 3, 4, 10, 15], "provid": [2, 3, 4, 5, 6, 7, 8, 13, 15, 19, 20, 22, 23, 24, 25], "paramet": [2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 15, 16, 17, 18, 19, 20, 22, 23, 24, 25, 27, 28, 32, 33], "group": [2, 3, 4], "constrain": [2, 3, 4, 13, 15, 16, 19, 20, 22, 23, 24, 25], "return": [2, 3, 4, 5, 6, 7, 8, 10, 13, 15, 16, 17, 18, 19, 20, 22, 23, 24, 25, 27, 28, 31, 32, 33], "type": [2, 3, 4, 5, 6, 7, 8, 12, 13, 15, 16, 17, 18, 19, 20, 22, 23, 24, 25, 27, 28, 32, 33], "geometr": 3, "recommend": [3, 34], "harmon": 4, "xavier": 4, "glorot": 4, "output_scal": [5, 6, 7, 8, 13, 15, 16, 19, 20, 23], "grad_input_scal": [5, 7, 13, 15, 19, 20, 23], "assum": [5, 6, 7, 8], "two": [5, 10, 16, 18], "select": [5, 6, 7, 8], "onli": [5, 6, 7, 8, 10, 16, 27, 28, 31, 32], "chosen": [5, 6, 7, 8], "factor": [5, 6, 7, 8, 23, 26, 27, 28], "op": [5, 6, 7, 8, 31, 32], "": [5, 6, 7, 8, 15, 16, 23, 31, 32, 33], "output": [5, 6, 7, 8, 10, 12, 15, 16, 19, 22, 24, 25, 31, 32, 33], "input": [5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 18, 19, 20, 22, 23, 24, 25, 27, 28, 31, 32, 33], "gradient": [5, 6, 7, 8, 10, 12, 13, 15, 16, 18, 19, 20, 22, 23, 24, 25, 31], "equal": [5, 6, 7, 8, 12, 18], "left_grad_scal": [6, 8, 16], "right_grad_scal": [6, 8, 16], "three": [6, 8, 16], "left": [6, 8, 16], "right": [6, 8, 16], "an": [7, 11, 12, 16, 18, 19, 23, 30, 31, 32], "i": [7, 10, 11, 12, 13, 15, 16, 18, 19, 20, 22, 23, 24, 25, 31, 32, 34], "ani": [7, 15, 22, 24, 25, 31, 32, 33], "number": [7, 10, 12, 14, 15, 22, 24, 25], "grad": [7, 19, 22, 24, 25, 27, 28, 31], "version": [9, 16, 19, 21], "torch": [9, 10, 12, 13, 15, 16, 19, 20, 21, 23, 24, 25, 31, 33], "nn": [9, 12, 21, 23, 24, 25, 31, 32, 33], "tensor": [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 27, 28, 30, 31, 32, 33], "target": [10, 32], "weight": [10, 12, 14, 15, 17, 18, 24, 25, 33], "option": [10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 23, 24, 25, 30, 32, 33], "none": [10, 12, 13, 14, 15, 16, 19, 20, 22, 23, 24, 25, 30, 31, 32, 33], "size_averag": 10, "bool": [10, 11, 12, 31, 32, 33], "ignore_index": 10, "int": [10, 12, 14, 20, 22, 23, 24, 25, 31, 32], "100": 10, "reduc": 10, "reduct": 10, "str": [10, 32, 33], "label_smooth": 10, "0": [10, 11, 12, 13, 17, 18, 20, 24, 25, 31, 33], "cross": 10, "entropi": 10, "loss": 10, "between": [10, 18], "logit": 10, "see": [10, 11, 12, 13, 14, 20, 25, 31, 32, 34], "crossentropyloss": 10, "detail": [10, 11, 12, 14, 20, 31, 32, 34], "predict": 10, "unnorm": 10, "shape": [10, 12, 15, 31], "section": 10, "below": 10, "support": [10, 11, 12, 15, 16, 31, 34], "ground": 10, "truth": 10, "class": [10, 20, 21, 22, 23, 24, 25, 29, 30, 31, 32, 33], "indic": [10, 12, 31], "probabl": [10, 11, 22, 24, 25], "manual": 10, "rescal": 10, "given": [10, 12, 31, 33], "each": [10, 12, 31, 32, 33], "If": [10, 11, 12, 13, 15, 16, 19, 20, 22, 23, 24, 25, 31, 32, 34], "ha": [10, 16, 31], "size": [10, 12, 22, 23, 24, 25], "c": [10, 31, 34], "deprec": 10, "By": 10, "default": [10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 22, 23, 24, 25, 31, 33], "ar": [10, 16, 18, 31], "averag": 10, "over": 10, "element": [10, 11, 13, 19, 20], "batch": [10, 12, 16], "note": [10, 12, 16, 24, 31, 34], "some": [10, 15, 16], "multipl": [10, 16, 27, 28], "per": 10, "sampl": [10, 11, 12], "field": 10, "set": [10, 11, 31, 34], "fals": [10, 11, 12, 31, 33], "instead": 10, "sum": [10, 20], "minibatch": 10, "ignor": 10, "when": [10, 13, 16, 32], "true": [10, 11, 12, 31, 32, 33], "specifi": [10, 12, 20], "valu": [10, 31, 32], "doe": [10, 16], "contribut": [10, 12], "non": [10, 12, 16, 31], "applic": [10, 31], "contain": [10, 12, 30], "observ": 10, "depend": [10, 16, 32], "appli": [10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 22, 23, 24, 25, 27, 28, 31], "divid": 10, "process": [10, 32], "being": [10, 18, 32], "meantim": 10, "either": [10, 16, 31], "those": 10, "arg": [10, 19, 31, 32], "overrid": 10, "A": [10, 12, 16, 20, 22, 23, 24, 25, 34], "1": [10, 12, 13, 16, 20, 31, 33], "amount": 10, "smooth": 10, "where": [10, 12, 13, 15, 16], "The": [10, 12, 16, 18, 31, 32], "becom": 10, "mixtur": 10, "origin": [10, 28, 32], "uniform": 10, "distribut": [10, 11, 13], "describ": 10, "rethink": 10, "incept": 10, "architectur": 10, "vision": 10, "n": [10, 16, 32], "d_1": 10, "d_2": 10, "d_k": 10, "k": [10, 16], "geq": 10, "case": 10, "dimension": [10, 16], "should": [10, 17, 18, 19, 31, 32, 33], "same": [10, 16, 31], "begin": 10, "align": 10, "text": [10, 13, 20], "end": 10, "exampl": [10, 12, 16, 31, 32, 33], "randn": [10, 33], "3": [10, 12, 13], "5": [10, 11, 12, 13], "requires_grad": [10, 31], "randint": 10, "dtype": [10, 15, 16, 20, 31, 32], "int64": 10, "f": [10, 12, 19], "backward": [10, 16, 18, 26, 27, 30, 31, 32, 33], "softmax": [10, 22, 24, 25], "dim": [10, 20], "p": [11, 12, 16], "train": [11, 12, 34], "inplac": [11, 31, 33], "from": [11, 31, 32, 33], "bernoulli": 11, "zero": [11, 31], "do": [11, 12, 31, 32], "thi": [11, 12, 15, 16, 18, 20, 24, 31, 32, 34], "place": [11, 12, 31], "padding_idx": 12, "max_norm": 12, "norm_typ": 12, "2": [12, 13, 15, 16, 17, 18, 24, 25, 31, 33], "scale_grad_by_freq": 12, "spars": [12, 15, 16], "lookup": 12, "tabl": 12, "look": [12, 16, 32], "up": [12, 32], "fix": 12, "dictionaryand": 12, "often": [12, 16], "retriev": [12, 31, 32], "word": 12, "list": [12, 32], "matrix": [12, 16], "correspond": [12, 31], "more": [12, 20, 31], "longtensor": 12, "row": 12, "maximum": 12, "possibl": 12, "index": [12, 34], "column": 12, "entri": 12, "therefor": 12, "vector": [12, 16], "updat": 12, "dure": [12, 31], "e": [12, 16, 24, 31, 32, 34], "remain": 12, "pad": 12, "norm": 12, "larger": 12, "than": [12, 32], "renorm": 12, "have": [12, 15, 16, 31], "modifi": [12, 31], "invers": 12, "frequenc": 12, "mini": 12, "w": [12, 31], "r": [12, 31, 34], "t": [12, 15, 31], "under": [12, 34], "regard": 12, "arbitrari": [12, 30], "extract": [12, 32], "point": 12, "v": 12, "embedding_dim": 12, "4": [12, 23, 31, 33], "9": 12, "10": [12, 33], "embedding_matrix": 12, "rand": 12, "xdoctest": [12, 31], "ignore_w": 12, "determinist": 12, "8490": 12, "9625": 12, "6753": 12, "9666": 12, "7761": 12, "6108": 12, "6246": 12, "9751": 12, "3618": 12, "4161": 12, "2419": 12, "7383": 12, "0237": 12, "7794": 12, "0528": 12, "3385": 12, "8612": 12, "1867": 12, "zero_": 12, "0000": 12, "5609": 12, "5384": 12, "8720": 12, "6262": 12, "2438": 12, "7471": 12, "constraint": [13, 15, 16, 19, 20, 22, 23, 24, 25, 32], "callabl": [13, 15, 16, 19, 20, 22, 23, 24, 25, 32, 33], "gmean": [13, 15, 16, 19, 20, 22, 23, 24, 25], "approxim": [13, 23, 24, 25], "argument": [13, 16, 31, 32], "wise": [13, 19], "x": [13, 31, 33], "phi": 13, "cumul": 13, "gaussian": 13, "tanh": 13, "estim": 13, "math": [13, 33], "sqrt": 13, "pi": 13, "044715": 13, "error": [13, 31, 32], "linear": [13, 33], "param": 13, "whichtak": [13, 15, 19, 20, 23], "singl": [13, 15, 16, 19, 20, 22, 23, 24, 25], "usual": [13, 15, 19, 20, 23], "necessari": [13, 15, 16, 18, 19, 20, 23, 31], "valid": [13, 15, 16, 19, 20, 23], "binaryconstraint": [13, 15, 19, 20, 23], "normalized_shap": 14, "sequenc": 14, "bia": [14, 15, 33], "ep": 14, "1e": 14, "05": 14, "layer": [14, 18, 22, 23, 24, 25], "normal": 14, "last": 14, "certain": [14, 16, 32], "dimens": [14, 15, 16, 20, 22, 23, 24, 25], "layernorm": 14, "transform": [15, 19, 24, 25], "incom": 15, "data": [15, 20, 32], "y": [15, 31], "xa": 15, "b": [15, 16, 31], "opper": 15, "d": [15, 31, 33], "layout": [15, 16, 32], "beta": [15, 16], "featur": [15, 16], "devic": [15, 16, 32], "combin": [15, 16, 17], "mai": [15, 16, 31], "autograd": [15, 16, 31], "you": [15, 16, 31], "notic": [15, 16, 34], "miss": [15, 16], "pleas": [15, 16], "open": [15, 16], "request": [15, 16], "tensorfloat32": [15, 16], "_featur": 15, "addit": 15, "includ": 15, "out": [15, 16, 17, 31, 34], "base": [15, 31, 34], "union": [16, 32, 33], "tupl": [16, 18, 31, 32, 33], "product": 16, "behavior": 16, "follow": [16, 19, 31], "both": [16, 33], "dot": 16, "scalar": [16, 27, 28, 30], "first": [16, 19, 31, 32, 34], "second": 16, "prepend": 16, "its": [16, 19, 28, 31], "purpos": 16, "multipli": 16, "after": [16, 31], "remov": 16, "least": 16, "one": [16, 31], "append": 16, "broadcast": 16, "thu": 16, "must": [16, 31], "For": 16, "j": 16, "time": [16, 34], "other": 16, "logic": 16, "determin": 16, "m": [16, 34], "even": 16, "though": 16, "final": 16, "differ": [16, 26, 34], "In": [16, 31], "particular": 16, "restrict": 16, "mm": 16, "On": 16, "rocm": 16, "float16": 16, "precis": [16, 34], "which": [16, 18, 20, 22, 23, 24, 25, 32], "take": [16, 19, 22, 24, 25], "order": [16, 18, 32], "atupl": 16, "expectedthat": 16, "all": [16, 20, 31], "isprovid": 16, "residu": [17, 18, 24, 25], "skip": [17, 18, 24, 25, 31], "tau": [17, 18, 24, 25], "add": [17, 34], "connect": [17, 18, 24, 25], "togeth": 17, "rel": [17, 18, 23, 24, 25], "branch": [17, 18, 24, 25], "conjunct": [17, 18], "residual_split": 17, "come": 17, "result": [17, 18, 32], "split": 18, "prior": [18, 31], "residual_add": 18, "delai": 18, "pass": [18, 26, 27, 28, 30, 31, 32, 33], "still": [18, 31], "need": [18, 31, 33], "normalis": 18, "standard": [18, 30, 31, 32, 33], "network": 18, "typic": 18, "kwarg": [19, 31, 32], "defin": [20, 31], "x_": 20, "frac": 20, "exp": 20, "x_i": 20, "sum_j": 20, "x_j": 20, "It": [20, 31], "slice": [20, 32], "along": 20, "re": 20, "them": [20, 31, 32], "so": [20, 31, 32], "lie": 20, "rang": 20, "desir": 20, "cast": 20, "befor": [20, 31, 32], "perform": [20, 31], "prevent": [20, 31], "overflow": 20, "hidden_s": [22, 23, 24, 25, 33], "head": [22, 24, 25], "dropout_p": [22, 24, 25], "implement": [22, 23, 24, 25, 33], "multi": 22, "self": [22, 32, 33], "attent": [22, 24, 25], "warn": [22, 24, 25], "here": [22, 24, 25], "like": [22, 24, 25, 31], "give": [22, 24, 25], "incorrect": [22, 24, 25, 31], "hidden": [22, 23, 24, 25], "post": [22, 24, 25], "dropout": [22, 24, 25], "act_fn": [23, 24, 25], "gelu": [23, 24, 25], "expansion_factor": 23, "activ": [23, 24, 25, 34], "intermedi": [23, 32], "increas": [23, 31], "vocab_s": 24, "decod": 24, "current": [24, 32], "just": [24, 31, 32], "demonstr": 24, "lack": 24, "kei": 24, "g": [24, 31, 32, 34], "causal": 24, "mask": 24, "posit": [24, 32], "embed": 24, "usag": 24, "infer": 24, "token": 24, "vocabulari": 24, "prenorm": 25, "http": 25, "arxiv": 25, "org": 25, "ab": 25, "2002": 25, "04745": 25, "enabl": [26, 31, 34], "forward": [26, 27, 28, 30, 31, 32, 33], "unchang": 27, "develop": 29, "model": 29, "dataclass": 30, "pair": 30, "intend": [30, 31], "repres": [30, 31, 32, 33], "deviat": [30, 31, 32, 33], "record": [31, 32], "suppli": 31, "dict": [31, 32], "static": 31, "ctx": 31, "functionctx": 31, "formula": 31, "differenti": 31, "mode": 31, "automat": [31, 33], "alia": 31, "vjp": 31, "overridden": 31, "subclass": 31, "accept": 31, "context": 31, "mani": 31, "were": 31, "requir": [31, 34], "can": [31, 32], "save": [31, 34], "also": 31, "attribut": [31, 32], "needs_input_grad": 31, "boolean": 31, "whether": 31, "jvp": 31, "grad_input": 31, "got": 31, "respect": 31, "object": 31, "mark_dirti": 31, "mark": 31, "call": [31, 32, 33], "most": 31, "onc": 31, "insid": 31, "method": [31, 32, 33], "everi": [31, 32], "been": 31, "ensur": 31, "correct": 31, "our": 31, "check": [31, 34], "doesn": 31, "matter": 31, "modif": 31, "staticmethod": 31, "def": [31, 33], "x_npy": 31, "numpi": 31, "share": 31, "storag": 31, "once_differenti": 31, "grad_output": 31, "doubl": 31, "clone": 31, "would": 31, "lead": 31, "wrong": 31, "engin": 31, "know": 31, "unless": 31, "we": [31, 32], "runtimeerror": 31, "variabl": 31, "mark_non_differenti": 31, "effici": 31, "alwai": 31, "go": 31, "sort": 31, "func": 31, "idx": 31, "save_for_backward": 31, "g1": 31, "g2": 31, "saved_tensor": 31, "zeros_lik": 31, "index_add_": 31, "futur": 31, "oppos": 31, "directli": 31, "memori": 31, "leak": 31, "hook": 31, "graph": [31, 32], "saved_tensors_hook": 31, "intermediari": 31, "neither": 31, "nor": 31, "your": [31, 34], "custom": 31, "decor": 31, "rais": 31, "recomput": 31, "tutori": 31, "access": 31, "through": [31, 32], "user": 31, "made": 31, "thei": 31, "weren": 31, "content": 31, "extend": 31, "how": 31, "z": 31, "grad_out": 31, "gx": 31, "gy": 31, "gz": 31, "save_for_forward": 31, "x_t": 31, "y_t": 31, "_": 31, "fwad": 31, "dual_level": 31, "a_dual": 31, "make_du": 31, "set_materialize_grad": 31, "materi": 31, "undefin": 31, "expand": 31, "full": 31, "simplefunc": 31, "No": 31, "handl": [31, 32], "now": 31, "induc": 31, "graphmodul": 32, "wrap": [32, 33], "fx": [32, 33], "execut": 32, "instrument": 32, "call_funct": 32, "node": 32, "semant": 32, "invoc": 32, "keyword": 32, "compat": 32, "api": 32, "guarante": 32, "call_method": 32, "complex": 32, "memory_format": 32, "call_modul": 32, "fetch_args_kwargs_from_env": 32, "fetch": 32, "concret": 32, "environ": 32, "fetch_attr": 32, "hierarchi": 32, "fulli": 32, "qualfii": 32, "name": 32, "get_attr": 32, "Will": 32, "wa": 32, "map_nodes_to_valu": 32, "recurs": [32, 33], "descend": 32, "structur": 32, "within": 32, "belong": 32, "report": 32, "realli": 32, "referenc": 32, "placehold": 32, "To": 32, "tracer": 32, "store": 32, "target_to_funct": 32, "run": [32, 34], "initial_env": 32, "enable_io_process": 32, "via": 32, "interpret": 32, "start": 32, "map": 32, "pre": [32, 34], "popul": 32, "partial": 32, "evalu": 32, "process_input": 32, "process_output": 32, "run_nod": 32, "specif": 32, "recurse_modul": 33, "syntax_highlight": 33, "autowrap_modul": 33, "built": 33, "einop": 33, "usr": 33, "local": 33, "lib": 33, "python3": [33, 34], "8": 33, "dist": 33, "packag": 33, "__init__": 33, "py": 33, "__w": 33, "autowrap_funct": 33, "dummi": 33, "gener": 33, "code": 33, "annot": 33, "analys": 33, "fed": 33, "analysi": 33, "equival": 33, "plain": 33, "toggl": 33, "behavour": 33, "moduletyp": 33, "u": 33, "python": [33, 34], "whose": 33, "without": 33, "string": 33, "reflect": 33, "mlp": 33, "super": 33, "fc1": 33, "relu": 33, "fc2": 33, "requires_grad_": 33, "bwd": 33, "print": 33, "236": 33, "fc1_weight": 33, "018": 33, "6": 33, "54": 33, "fc1_bia": 33, "0182": 33, "51": 33, "_c": 33, "_nn": 33, "578": 33, "204": 33, "337": 33, "288": 33, "fc2_weight": 33, "00902": 33, "13": 33, "fc2_bia": 33, "00904": 33, "31": 33, "linear_1": 33, "235": 33, "999": 33, "librari": 34, "pytorch": 34, "paper": 34, "box": 34, "low": 34, "setup": 34, "venv": 34, "bin": 34, "path_to_poplar_sdk": 34, "ipu": 34, "pip": 34, "instal": 34, "wheel": 34, "poplar_sdk_en": 34, "poptorch": 34, "whl": 34, "dev": 34, "txt": 34, "subsequ": 34, "flight": 34, "help": 34, "command": 34, "id": 34, "intepret": 34, "format": 34, "consid": 34, "env": 34, "file": 34, "pythonpath": 34, "echo": 34, "pwd": 34, "path": 34, "devcontain": 34, "doc": 34, "cd": 34, "make": 34, "html": 34, "view": 34, "_build": 34, "browser": 34, "copyright": 34, "2023": 34, "graphcor": 34, "ltd": 34, "mit": 34, "md": 34, "further": 34}, "objects": {"": [[0, 0, 0, "-", "unit_scaling"]], "unit_scaling": [[1, 0, 0, "-", "constraints"], [9, 0, 0, "-", "functional"], [21, 0, 0, "-", "modules"], [26, 0, 0, "-", "scale"], [29, 0, 0, "-", "utils"]], "unit_scaling.constraints": [[2, 1, 1, "", "amean"], [3, 1, 1, "", "gmean"], [4, 1, 1, "", "hmean"], [5, 1, 1, "", "to_grad_input_scale"], [6, 1, 1, "", "to_left_grad_scale"], [7, 1, 1, "", "to_output_scale"], [8, 1, 1, "", "to_right_grad_scale"]], "unit_scaling.functional": [[10, 1, 1, "", "cross_entropy"], [11, 1, 1, "", "dropout"], [12, 1, 1, "", "embedding"], [13, 1, 1, "", "gelu"], [14, 1, 1, "", "layer_norm"], [15, 1, 1, "", "linear"], [16, 1, 1, "", "matmul"], [17, 1, 1, "", "residual_add"], [18, 1, 1, "", "residual_split"], [19, 1, 1, "", "scale_elementwise"], [20, 1, 1, "", "softmax"]], "unit_scaling.modules": [[22, 2, 1, "", "MHSA"], [23, 2, 1, "", "MLP"], [24, 2, 1, "", "TransformerDecoder"], [25, 2, 1, "", "TransformerLayer"]], "unit_scaling.scale": [[27, 1, 1, "", "scale_bwd"], [28, 1, 1, "", "scale_fwd"]], "unit_scaling.utils": [[30, 2, 1, "", "ScalePair"], [31, 2, 1, "", "ScaleTracker"], [32, 2, 1, "", "ScaleTrackingInterpreter"], [33, 1, 1, "", "analyse_module"]], "unit_scaling.utils.ScaleTracker": [[31, 3, 1, "", "backward"], [31, 3, 1, "", "jvp"], [31, 3, 1, "", "mark_dirty"], [31, 3, 1, "", "mark_non_differentiable"], [31, 3, 1, "", "save_for_backward"], [31, 3, 1, "", "save_for_forward"], [31, 3, 1, "", "set_materialize_grads"], [31, 3, 1, "", "vjp"]], "unit_scaling.utils.ScaleTrackingInterpreter": [[32, 3, 1, "", "call_function"], [32, 3, 1, "", "call_method"], [32, 3, 1, "", "call_module"], [32, 3, 1, "", "fetch_args_kwargs_from_env"], [32, 3, 1, "", "fetch_attr"], [32, 3, 1, "", "get_attr"], [32, 3, 1, "", "map_nodes_to_values"], [32, 3, 1, "", "output"], [32, 3, 1, "", "placeholder"], [32, 3, 1, "", "run"], [32, 3, 1, "", "run_node"]]}, "objtypes": {"0": "py:module", "1": "py:function", "2": "py:class", "3": "py:method"}, "objnames": {"0": ["py", "module", "Python module"], "1": ["py", "function", "Python function"], "2": ["py", "class", "Python class"], "3": ["py", "method", "Python method"]}, "titleterms": {"unit_sc": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33], "constraint": [1, 2, 3, 4, 5, 6, 7, 8], "amean": 2, "gmean": 3, "hmean": 4, "to_grad_input_scal": 5, "to_left_grad_scal": 6, "to_output_scal": 7, "to_right_grad_scal": 8, "function": [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], "cross_entropi": 10, "dropout": 11, "embed": 12, "gelu": 13, "layer_norm": 14, "linear": 15, "matmul": 16, "residual_add": 17, "residual_split": 18, "scale_elementwis": 19, "softmax": 20, "modul": [21, 22, 23, 24, 25], "mhsa": 22, "mlp": 23, "transformerdecod": 24, "transformerlay": 25, "scale": [26, 27, 28, 34], "scale_bwd": 27, "scale_fwd": 28, "util": [29, 30, 31, 32, 33], "scalepair": 30, "scaletrack": 31, "scaletrackinginterpret": 32, "analyse_modul": 33, "unit": 34, "develop": 34, "licens": 34, "api": 34}, "envversion": {"sphinx.domains.c": 2, "sphinx.domains.changeset": 1, "sphinx.domains.citation": 1, "sphinx.domains.cpp": 8, "sphinx.domains.index": 1, "sphinx.domains.javascript": 2, "sphinx.domains.math": 2, "sphinx.domains.python": 3, "sphinx.domains.rst": 2, "sphinx.domains.std": 2, "sphinx.ext.intersphinx": 1, "sphinx.ext.viewcode": 1, "sphinx": 57}, "alltitles": {"unit_scaling": [[0, "module-unit_scaling"]], "unit_scaling.constraints": [[1, "module-unit_scaling.constraints"]], "unit_scaling.constraints.amean": [[2, "unit-scaling-constraints-amean"]], "unit_scaling.constraints.gmean": [[3, "unit-scaling-constraints-gmean"]], "unit_scaling.constraints.hmean": [[4, "unit-scaling-constraints-hmean"]], "unit_scaling.constraints.to_grad_input_scale": [[5, "unit-scaling-constraints-to-grad-input-scale"]], "unit_scaling.constraints.to_left_grad_scale": [[6, "unit-scaling-constraints-to-left-grad-scale"]], "unit_scaling.constraints.to_output_scale": [[7, "unit-scaling-constraints-to-output-scale"]], "unit_scaling.constraints.to_right_grad_scale": [[8, "unit-scaling-constraints-to-right-grad-scale"]], "unit_scaling.functional": [[9, "module-unit_scaling.functional"]], "unit_scaling.functional.cross_entropy": [[10, "unit-scaling-functional-cross-entropy"]], "unit_scaling.functional.dropout": [[11, "unit-scaling-functional-dropout"]], "unit_scaling.functional.embedding": [[12, "unit-scaling-functional-embedding"]], "unit_scaling.functional.gelu": [[13, "unit-scaling-functional-gelu"]], "unit_scaling.functional.layer_norm": [[14, "unit-scaling-functional-layer-norm"]], "unit_scaling.functional.linear": [[15, "unit-scaling-functional-linear"]], "unit_scaling.functional.matmul": [[16, "unit-scaling-functional-matmul"]], "unit_scaling.functional.residual_add": [[17, "unit-scaling-functional-residual-add"]], "unit_scaling.functional.residual_split": [[18, "unit-scaling-functional-residual-split"]], "unit_scaling.functional.scale_elementwise": [[19, "unit-scaling-functional-scale-elementwise"]], "unit_scaling.functional.softmax": [[20, "unit-scaling-functional-softmax"]], "unit_scaling.modules": [[21, "module-unit_scaling.modules"]], "unit_scaling.modules.MHSA": [[22, "unit-scaling-modules-mhsa"]], "unit_scaling.modules.MLP": [[23, "unit-scaling-modules-mlp"]], "unit_scaling.modules.TransformerDecoder": [[24, "unit-scaling-modules-transformerdecoder"]], "unit_scaling.modules.TransformerLayer": [[25, "unit-scaling-modules-transformerlayer"]], "unit_scaling.scale": [[26, "module-unit_scaling.scale"]], "unit_scaling.scale.scale_bwd": [[27, "unit-scaling-scale-scale-bwd"]], "unit_scaling.scale.scale_fwd": [[28, "unit-scaling-scale-scale-fwd"]], "unit_scaling.utils": [[29, "module-unit_scaling.utils"]], "unit_scaling.utils.ScalePair": [[30, "unit-scaling-utils-scalepair"]], "unit_scaling.utils.ScaleTracker": [[31, "unit-scaling-utils-scaletracker"]], "unit_scaling.utils.ScaleTrackingInterpreter": [[32, "unit-scaling-utils-scaletrackinginterpreter"]], "unit_scaling.utils.analyse_module": [[33, "unit-scaling-utils-analyse-module"]], "Unit Scaling": [[34, "unit-scaling"]], "Development": [[34, "development"]], "License": [[34, "license"]], "API": [[34, "api"]]}, "indexentries": {"module": [[0, "module-unit_scaling"], [1, "module-unit_scaling.constraints"], [9, "module-unit_scaling.functional"], [21, "module-unit_scaling.modules"], [26, "module-unit_scaling.scale"], [29, "module-unit_scaling.utils"]], "unit_scaling": [[0, "module-unit_scaling"]], "unit_scaling.constraints": [[1, "module-unit_scaling.constraints"]], "amean() (in module unit_scaling.constraints)": [[2, "unit_scaling.constraints.amean"]], "gmean() (in module unit_scaling.constraints)": [[3, "unit_scaling.constraints.gmean"]], "hmean() (in module unit_scaling.constraints)": [[4, "unit_scaling.constraints.hmean"]], "to_grad_input_scale() (in module unit_scaling.constraints)": [[5, "unit_scaling.constraints.to_grad_input_scale"]], "to_left_grad_scale() (in module unit_scaling.constraints)": [[6, "unit_scaling.constraints.to_left_grad_scale"]], "to_output_scale() (in module unit_scaling.constraints)": [[7, "unit_scaling.constraints.to_output_scale"]], "to_right_grad_scale() (in module unit_scaling.constraints)": [[8, "unit_scaling.constraints.to_right_grad_scale"]], "unit_scaling.functional": [[9, "module-unit_scaling.functional"]], "cross_entropy() (in module unit_scaling.functional)": [[10, "unit_scaling.functional.cross_entropy"]], "dropout() (in module unit_scaling.functional)": [[11, "unit_scaling.functional.dropout"]], "embedding() (in module unit_scaling.functional)": [[12, "unit_scaling.functional.embedding"]], "gelu() (in module unit_scaling.functional)": [[13, "unit_scaling.functional.gelu"]], "layer_norm() (in module unit_scaling.functional)": [[14, "unit_scaling.functional.layer_norm"]], "linear() (in module unit_scaling.functional)": [[15, "unit_scaling.functional.linear"]], "matmul() (in module unit_scaling.functional)": [[16, "unit_scaling.functional.matmul"]], "residual_add() (in module unit_scaling.functional)": [[17, "unit_scaling.functional.residual_add"]], "residual_split() (in module unit_scaling.functional)": [[18, "unit_scaling.functional.residual_split"]], "scale_elementwise() (in module unit_scaling.functional)": [[19, "unit_scaling.functional.scale_elementwise"]], "softmax() (in module unit_scaling.functional)": [[20, "unit_scaling.functional.softmax"]], "unit_scaling.modules": [[21, "module-unit_scaling.modules"]], "mhsa (class in unit_scaling.modules)": [[22, "unit_scaling.modules.MHSA"]], "mlp (class in unit_scaling.modules)": [[23, "unit_scaling.modules.MLP"]], "transformerdecoder (class in unit_scaling.modules)": [[24, "unit_scaling.modules.TransformerDecoder"]], "transformerlayer (class in unit_scaling.modules)": [[25, "unit_scaling.modules.TransformerLayer"]], "unit_scaling.scale": [[26, "module-unit_scaling.scale"]], "scale_bwd() (in module unit_scaling.scale)": [[27, "unit_scaling.scale.scale_bwd"]], "scale_fwd() (in module unit_scaling.scale)": [[28, "unit_scaling.scale.scale_fwd"]], "unit_scaling.utils": [[29, "module-unit_scaling.utils"]], "scalepair (class in unit_scaling.utils)": [[30, "unit_scaling.utils.ScalePair"]], "scaletracker (class in unit_scaling.utils)": [[31, "unit_scaling.utils.ScaleTracker"]], "backward() (unit_scaling.utils.scaletracker static method)": [[31, "unit_scaling.utils.ScaleTracker.backward"]], "jvp() (unit_scaling.utils.scaletracker static method)": [[31, "unit_scaling.utils.ScaleTracker.jvp"]], "mark_dirty() (unit_scaling.utils.scaletracker method)": [[31, "unit_scaling.utils.ScaleTracker.mark_dirty"]], "mark_non_differentiable() (unit_scaling.utils.scaletracker method)": [[31, "unit_scaling.utils.ScaleTracker.mark_non_differentiable"]], "save_for_backward() (unit_scaling.utils.scaletracker method)": [[31, "unit_scaling.utils.ScaleTracker.save_for_backward"]], "save_for_forward() (unit_scaling.utils.scaletracker method)": [[31, "unit_scaling.utils.ScaleTracker.save_for_forward"]], "set_materialize_grads() (unit_scaling.utils.scaletracker method)": [[31, "unit_scaling.utils.ScaleTracker.set_materialize_grads"]], "vjp() (unit_scaling.utils.scaletracker static method)": [[31, "unit_scaling.utils.ScaleTracker.vjp"]], "scaletrackinginterpreter (class in unit_scaling.utils)": [[32, "unit_scaling.utils.ScaleTrackingInterpreter"]], "call_function() (unit_scaling.utils.scaletrackinginterpreter method)": [[32, "unit_scaling.utils.ScaleTrackingInterpreter.call_function"]], "call_method() (unit_scaling.utils.scaletrackinginterpreter method)": [[32, "unit_scaling.utils.ScaleTrackingInterpreter.call_method"]], "call_module() (unit_scaling.utils.scaletrackinginterpreter method)": [[32, "unit_scaling.utils.ScaleTrackingInterpreter.call_module"]], "fetch_args_kwargs_from_env() (unit_scaling.utils.scaletrackinginterpreter method)": [[32, "unit_scaling.utils.ScaleTrackingInterpreter.fetch_args_kwargs_from_env"]], "fetch_attr() (unit_scaling.utils.scaletrackinginterpreter method)": [[32, "unit_scaling.utils.ScaleTrackingInterpreter.fetch_attr"]], "get_attr() (unit_scaling.utils.scaletrackinginterpreter method)": [[32, "unit_scaling.utils.ScaleTrackingInterpreter.get_attr"]], "map_nodes_to_values() (unit_scaling.utils.scaletrackinginterpreter method)": [[32, "unit_scaling.utils.ScaleTrackingInterpreter.map_nodes_to_values"]], "output() (unit_scaling.utils.scaletrackinginterpreter method)": [[32, "unit_scaling.utils.ScaleTrackingInterpreter.output"]], "placeholder() (unit_scaling.utils.scaletrackinginterpreter method)": [[32, "unit_scaling.utils.ScaleTrackingInterpreter.placeholder"]], "run() (unit_scaling.utils.scaletrackinginterpreter method)": [[32, "unit_scaling.utils.ScaleTrackingInterpreter.run"]], "run_node() (unit_scaling.utils.scaletrackinginterpreter method)": [[32, "unit_scaling.utils.ScaleTrackingInterpreter.run_node"]], "analyse_module() (in module unit_scaling.utils)": [[33, "unit_scaling.utils.analyse_module"]]}})