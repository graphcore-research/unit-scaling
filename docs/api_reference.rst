API Reference
=============

:code:`unit-scaling` is implemented using thin wrappers around existing :code:`torch.nn`
classes and functions. Documentation also inherits from the standard PyTorch docs, with
modifications for scaling. Note that some docs may no longer be relevant but are
nevertheless inherited.

The API is built to mirror :code:`torch.nn` as closely as possible, such that PyTorch
classes and functions can easily be swapped-out for their unit-scaled equivalents.

For PyTorch code which uses the following imports:

.. code-block::

   from torch import nn
   from torch import functional as F

unit scaling can be applied by first adding:

.. code-block::

   import unit_scaling as uu
   from unit_scaling import functional as U

and then replacing the letters :code:`nn` with :code:`uu` and
:code:`F` with :code:`U`, for those classes/functions to be unit-scaled
(assuming they are supported).

Click below for the full documentation:

.. autosummary::
   :toctree: generated
   :template: custom-module-template.rst
   :recursive:

   unit_scaling