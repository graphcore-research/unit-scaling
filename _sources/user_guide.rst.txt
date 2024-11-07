User guide
==========

We recommend that new users read this guide before attempting to implement a unit-scaled
model.

It covers a brief overview of the technique, a practical guide to using the
library, some key considerations when applying unit scaling, and a discussion of
optimising unit-scaled models.

.. Note:: The library is currently in its *beta* release.
    Some features have yet to be implemented and occasional bugs may be present.
    We're keen to help users with any problems they encounter.

Installation
------------

To install the :code:`unit-scaling` library, run:

.. code-block::

    pip install unit-scaling

For those who wish to develop on the :code:`unit-scaling` codebase, clone or fork our
`GitHub repo <https://github.com/graphcore-research/unit-scaling.git>`_ and follow the
instructions in our :doc:`developer guide <development>`.

What is unit scaling?
---------------------

Unit scaling is a paradigm for designing deep learning models that aims to scale all
tensors (weights, activations and gradients) so that their standard deviation is
approximately 1 for the first pass of model training, before any weight updates have
taken place. This can enable the use of low-precision number formats out-of-the-box.

"Scaling" simply involves multiplying the output of an operation by a scalar value.
We use the term "scale" to refer to the standard deviation of a tensor.
Many operations used in deep learning change the scale of their inputs, and often in
an arbitrary way. This library is a re-implementation of common
PyTorch ops, adding scaling factors to ensure that input scale is now preserved.

As unit scaling is a technique for training models (usually from scratch), it considers
the scaling of operations in the backward pass as well as the forward pass.
The scaling factors used here are all pre-determined, based on
assumptions regarding the distribution of input tensors (typically, that they are
normally-distributed and unit-scaled).

The advantage of using a unit-scaled model is as follows:

1. A standard deviation of 1 is a great starting-point from the perspective of
   floating-point number formats. It gives roughly equal headroom for the scale to grow
   or shrink during training before over/underflow occur.
2. Because of this, loss scaling is not required for unit-scaled models.
   Although scales will drift from their unit starting-point during training,
   scales have stayed within range for all unit-scaled models tested thus far.
3. This can enable the use of smaller, more efficient number formats out-of-the-box,
   such as FP16 and even FP8.


How to unit-scale a model
-------------------------

We recommend the following approach to applying unit scaling to a model. We assume here
that you have an existing PyTorch model which you wish to adapt to be unit-scaled,
though a similar approach can be used to design a unit-scaled model from scratch.

**1. Consider your number formats**

The key motivation for unit scaling is to help keep values in the range of their number
formats. Given this, it makes sense to begin by understanding which values might go out
of range.

For those tensors in FP32 or BF16, range issues are unlikely to occur as these formats
can represent very large/small numbers (roughly 3e+38 to 1e-45).

Tensors in FP16 or FP8 are likely to require unit-scaling. FP16 and the FP8 E5
format can represent numbers between roughly 60,000 and 6e-05
(FP8 E4 has an even smaller range). Operations which use values in these formats may
require unit scaling.

We recommend that you try and put as many tensors as possible into low-precision formats as
this can speed up training considerably, and is where unit scaling is most useful.
A full discussion of which tensors should be in which format is beyond the scope of this
introduction.

**2. Analyse scaling**

The next step is to understand the scales present in the initial (non-unit-scaled)
model. This analysis can be tricky to implement, particularly in the backward pass, so
we provide a tool to make this analysis easier:
:py:func:`unit_scaling.utils.analyse_module`.

Using :external+pytorch:py:mod:`torch.fx`, this provides a line-by-line breakdown of a given model,
alongside the scale of the output tensor for each operation, in both the forward and
backward pass.

For example, given the following implementation of an MLP layer:

.. code-block::

    import torch
    import torch.nn.functional as F
    from torch import nn
    from unit_scaling.utils import analyse_module

    class UnscaledMLP(nn.Module):
        def __init__(self, d: int) -> None:
            super().__init__()
            self.linear_1 = nn.Linear(d, d * 4)
            self.linear_2 = nn.Linear(d * 4, d)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.linear_1(x)
            x = F.gelu(x)
            return self.linear_2(x)

we can use :py:func:`~unit_scaling.utils.analyse_module` to derive the following
analysis:

.. code-block::

    >>> x = torch.randn(2**8, 2**10).requires_grad_()  # fed into fwd pass
    >>> bwd = torch.randn(2**8, 2**10)  # fed into bwd pass
    >>> annotated_code = analyse_module(UnscaledMLP(2**10), x, bwd)
    >>> print(annotated_code)

    def forward(self, x : torch.Tensor) -> torch.Tensor:  (-> 1.0, <- 0.204)
        linear_1_weight = self.linear_1.weight;  (-> 0.018, <- 2.83)
        linear_1_bias = self.linear_1.bias;  (-> 0.018, <- 2.84)
        linear = torch._C._nn.linear(x, linear_1_weight, linear_1_bias);  (-> 0.578, <- 0.177)
        gelu = torch._C._nn.gelu(linear);  (-> 0.322, <- 0.289)
        linear_2_weight = self.linear_2.weight;  (-> 0.00902, <- 5.48)
        linear_2_bias = self.linear_2.bias;  (-> 0.00894, <- 16.1)
        linear_1 = torch._C._nn.linear(gelu, linear_2_weight, linear_2_bias);  (-> 0.198, <- 1.0)
        return linear_1

Firstly, :py:func:`~unit_scaling.utils.analyse_module` has decomposed the module into a set of low-level
operations. Secondly, it has appended each line with a tuple
:code:`(-> fwd_scale, <- bwd_scale)` denoting the scale of the tensor on the left of
the :code:`=` sign in the forward and backward passes.

We can see from the above example that this module is not well-scaled. In both passes
we begin with a scale of 1 (as this is what we fed in). By the end of the forward pass
the scale is 0.198, and by the end of the backward pass the scale is 0.204. Along the
way we generate large scales for some of the weight gradients, with
:code:`linear_2_bias` receiving a gradient of scale 16.1.

These scales are not large or small enough to be a problem for our number formats, but in a
full model the unscaled operations could cause more significant numerical issues.
We show below how to address this using unit scaling.

(note: :py:func:`~unit_scaling.utils.analyse_module` can't be used on a model wrapped in
:code:`torch.compile`)

**3. Swap in unit-scaled ops**

By swapping-in unit-scaled versions of the operations in the module, we can correct
these scaling factors. :code:`unit-scaling` provides drop-in replacements:

.. code-block::

    import unit_scaling as uu
    import unit_scaling.functional as U

    class ScaledMLP(nn.Module):
        def __init__(self, d: int) -> None:
            super().__init__()
            self.linear_1 = uu.Linear(d, d * 4)  # Changed `nn` to `uu`
            self.linear_2 = uu.Linear(d * 4, d)  # Changed `nn` to `uu`

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.linear_1(x)
            x = U.gelu(x)  # Changed `F` to `U`
            return self.linear_2(x)

.. code-block::

    >>> annotated_code = analyse_module(ScaledMLP(2**10), x, bwd)
    >>> print(annotated_code)

    def forward(self, x : torch.Tensor) -> torch.Tensor:  (-> 1.0, <- 1.01)
        linear_1_weight = self.linear_1.weight;  (-> 1.0, <- 0.716)
        linear_1_bias = self.linear_1.bias;  (-> 0.0, <- 0.729)
        linear = U.linear(x, linear_1_weight, linear_1_bias, gmean);  (-> 0.707, <- 0.716)
        gelu = U.gelu(linear);  (-> 0.64, <- 0.706)
        linear_2_weight = self.linear_2.weight;  (-> 1.0, <- 0.693)
        linear_2_bias = self.linear_2.bias;  (-> 0.0, <- 1.03)
        linear_1 = U.linear(gelu, linear_2_weight, linear_2_bias, gmean);  (-> 0.979, <- 0.999)
        return linear_1

Note that not all modules and functions are implemented in :code:`unit-scaling`.
Implementations of the basic operations required for a transformer are available, but
many other operations are not yet provided.

For the set of modules and functions currently implemented, see :numref:`API Reference`.

**4. Repeat steps 2 & 3 until scales look good**

It's important to check that swapping in unit-scaled ops has the desired effect on
the scales in a model. There may be cases in which this is not the case, and additional
measures are required.

Understanding when tensor scales are "good enough" is something of an art. Generally,
when the standard deviation begins to approach the max/min values defined by a format then
numerical issues arise. For overflow, this is typically seen clearly in the loss
exploding (even with gradient clipping). Conversely, underflow tends to cause the loss
to degrade more steadily.

It's not necessary to keep scales at exactly 1, and unit-scaling is designed to only
approximately meet this target. In practice, scales of between 1/10 to 10 are of no
concern and are to be expected. Significantly smaller or larger scales may merit further
investigation (particularly larger).

**5. Optimise**

To attain the best performance, we recommend models are wrapped in
:external+pytorch:py:func:`torch.compile` (requires PyTorch >=2.0).
This is enabled via:

.. code-block::

    class Model(torch.nn.Module)
        def __init__(self):
            ...
    
    model = torch.compile(Model())

or

.. code-block::

    @torch.compile
    class Model(torch.nn.Module)
        def __init__(self):
            ...

As outlined in the
`torch.compile <https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html>`_,
documentation, compilation is a general-purpose
optimisation for models. It's particularly useful in the case of unit scaling, in order
to fuse scaling factors with operations
(see :numref:`Optimising unit-scaled models` for more detail).

Key considerations for unit scaling
-----------------------------------

**Loss functions**

The most important operation in the model to unit-scale is the loss function.
The division term and log-softmax used in the standard cross-entropy loss tend to
shrink gradients substantially.
The implementation in :code:`unit_scaling` provides scaled versions of
:external+pytorch:py:func:`torch.nn.functional.cross_entropy`
and :external+pytorch:py:class:`torch.nn.CrossEntropyLoss`
which correct for this. We recommend that you start here when unit-scaling your models.

**Linear layers**

In non-unit-scaled models, linear layers have a mechanism for controlling the scale:
their initialisation. The standard Xavier/Glorot initialisation provides good scaling
for activations and their gradients by pushing a (small) scaling factor into the weights
themselves. However, it does not provide good scaling for weight gradients.

Unit scaling solves this problem by taking a different approach: keeping scaling factors
outside the weights, which then enables separate scaling factors for activation
gradients and weight gradients. Because of this, you should expect your weights
to begin with scale=1 when using :code:`unit_scaling`. Alternative weight
initialisations should not be used in conjunction with unit scaling.

**Residual layers**

Particular care must be taken when using residual connections in unit-scaled models.
We provide two methods for residual scaling, which must be used together.

Consider a PyTorch residual layer of the form:

.. code-block::

    class ResidualLayer(nn.Module):
        def __init__(self):
            self.f = ...

        def forward(self, x):
            skip = x
            residual = self.f(x)
            return residual + skip

The unit-scaled equivalent should be implemented as:

.. code-block::

    class ResidualLayer(nn.Module):
        def __init__(self, tau=0.5):
            self.f = ...
            self.tau = tau

        def forward(self, x):
            residual, skip = U.residual_split(x, self.tau)
            residual = self.f(residual)
            return U.residual_add(residual, skip, self.tau)

This step is necessary because unit-scaled models give equal scale to the skip and
residual connections. In contrast, non-unit-scaled models tend change the scale of
activations as they go through the residual connection, meaning that when the residual
connection is added to the skip connection the ratio of the two scales is not 50:50.

The :code:`tau` hyperparameter is a scale-factor applied to the residual branch to
correct for this. In practice you may be able to leave it at the default value of 0.5
without having to tune this as an additional hyperparameter.

However in the case of self-attention layers, we find that tau must be dropped to
approximately 0.01. The default of 0.5 (which weights the branches 50:50) causes
significant degradation. This reflects the fact that in standard transformers the
self-attention layer down-scales the residual branch. Note that for MLP layers the
default tau=0.5 is sufficient.

We also employ a trick to ensure that this scaling factor is delayed in the backward
pass to keep values unit-scaled along the residual branch in both passes
(see :py:func:`~unit_scaling.functional.residual_split` for further details).
A more comprehensive discussion of this feature can be found in the
`unit scaling paper
<https://arxiv.org/abs/2303.11257>`_.

**Constraints**

Many unit-scaled operations introduce a :code:`constraint: Callable` argument.
*In most cases, you can simply leave this argument to take the default value and ignore it.*

The purpose of this constraint is that in some scenarios,
particular scaling factors in the
forward and backward passes must all be identical in order to produce
valid gradients. This constraint argument specifies how to arrive at the shared scale.

For example, the implementation of :py:func:`unit_scaling.functional.linear` contains the
following code:

.. code-block::

    output_scale = fan_in**-0.5
    grad_input_scale = fan_out**-0.5
    grad_weight_scale = grad_bias_scale = batch_size**-0.5
    if constraint:
        output_scale = grad_input_scale = constraint(output_scale, grad_input_scale)

First the "ideal" output and input-gradient scales are computed, and are then combined
using the provided constraint (if one is supplied). Constraining these values to be
the same for a linear layer is necessary to ensure valid gradients. This can cause
deviations from exact unit-scale, but these tend not to be significant.

The default value of :code:`constraint` is typically
:py:func:`unit_scaling.constraints.gmean`
(the geometric mean), representing a compromise between the forward and backward passes.
Note that we don't need to constrain the weight scale as this is allowed to
differ from the output/input-grad scales.

The `unit scaling paper
<https://arxiv.org/abs/2303.11257>`_ provides a comprehensive overview of where and why
constraints are required.

Optimising unit-scaled models
-----------------------------

Unit scaling adds extra scalar multiplications to each operation.
By default, PyTorch's eager evaluation causes each of these multiplications to make an
additional trip to-and-from memory.

Fortunately, his overhead can be eliminated via *kernel fusion*
(see this `Stack Overflow answer <https://stackoverflow.com/a/53311373>`_
for more details). In PyTorch there are two ways of fusing operations.

The "old" method uses :external+pytorch:py:func:`torch.jit.script` to convert PyTorch into a TorchScript
program, which is then just-in-time compiled.
However, many models can't be converted to TorchScript directly.

To rectify this, PyTorch 2.0 introduced a new method: :external+pytorch:py:func:`torch.compile`.
This approach is much more flexible and in theory can work on
arbitrary PyTorch programs. It can be applied to functions or modules as follows:

.. code-block::

    @torch.compile
    def unit_scaled_function(x):
        ...
    
    @torch.compile
    class UnitScaledModule(torch.nn.Module):
        def __init__(self):
            ...

Please refer to the `torch.compile
tutorial <https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html>`_
for further details.

For unit scaling, :external+pytorch:py:func:`torch.compile` fuses scaling factors where possible in the
forward and backward passes. This removes the overhead incurred when naively
adding scaling factors without fusion
(see the
`benchmarking compiled unit-scaled ops <https://github.com/graphcore-research/unit-scaling/tree/main/analysis/benchmarking_compiled_unit_scaled_ops.ipynb>`_
notebook for a thorough analysis).

:code`unit-scaling` does not automatically apply
:external+pytorch:py:func:`torch.compile`, so users will have to do this manually.
We strongly recommend users consider doing so
in order to get the most substantial speedups,
ideally in large blocks or compiling the entire model.

Note that there's a bug in the latest PyTorch version (<= 2.0.1) meaning the backward pass
fails to fuse scaling factors. This has recently been addressed, but
users will need to upgrade to the
`Preview (Nightly) build <https://pytorch.org/get-started/locally/>`_ (until
PyTorch 2.0.2 is released) to get the fix.
