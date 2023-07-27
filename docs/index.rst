Unit Scaling
============

Welcome to the :code:`unit-scaling` library. This library is designed to facilitate
the use of the *unit scaling* method, as outlined in the paper
`Unit Scaling: Out-of-the-Box Low-Precision Training (ICML, 2023)
<https://arxiv.org/abs/2303.11257>`_.

Installation
------------

To install :code:`unit-scaling`, run:

.. code-block::

    pip install git+https://github.com/graphcore-research/unit-scaling.git

For those who wish to develop on the :code:`unit-scaling` codebase, clone or fork our
`GitHub repo <https://github.com/graphcore-research/unit-scaling.git>`_ and follow the
instructions in our :doc:`developer guide <development>`.

.. Note:: This library is currently in its *beta* release.
    Some features have yet to be implemented and occasional bugs may be present.
    We're keen to help users with any problems they encounter.

Getting Started
------------

The following video illustrates how unit scaling works.

.. raw:: html
   :file: _static/animation.html

We recommend that new users get started with :numref:`User Guide`.
A reference outlining our API can be found at :numref:`API reference`.

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
