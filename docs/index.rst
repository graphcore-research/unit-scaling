Unit Scaling
============

.. Warning:: this library is currently in its *alpha* release. This means it's
   missing some functionality and should not be expected to work seamlessly.
   We're keen to help early users with any problems they encounter.

Welcome to the :code:`unit-scaling` library. This library is designed to facilitate
the use of the *unit scaling* method, as outlined in the paper
`Unit Scaling: Out-of-the-Box Low-Precision Training (ICML, 2023)
<https://arxiv.org/abs/2303.11257>`_.

The following video illustrates how unit scaling works.

.. raw:: html
   :file: _static/animation.html

We recommend that new users get started with :numref:`User Guide`.

For a high-level overview of the technique, see the blog post
`Simple FP16 and FP8 training with Unit Scaling
<https://www.graphcore.ai/posts/simple-fp16-and-fp8-training-with-unit-scaling>`_.


.. toctree::
    :caption: Contents
    :numbered:
    :maxdepth: 3

    User guide <user_guide>
    Developer guide <development>
    Limitations <limitations>
    API reference <api_reference>
