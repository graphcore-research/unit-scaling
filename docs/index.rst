Unit Scaling
============

Welcome to the :code:`unit-scaling` library. This library is designed to facilitate
the use of the *unit scaling* method, as outlined in the paper
`Unit Scaling: Out-of-the-Box Low-Precision Training (ICML, 2023)
<https://arxiv.org/abs/2303.11257>`_.

For a demonstration of the library, see `Out-of-the-Box FP8 Training
<https://github.com/graphcore-research/out-of-the-box-fp8-training/blob/main/out_of_the_box_fp8_training.ipynb>`_ — a notebook showing how to unit-scale the nanoGPT model.

Installation
------------

To install :code:`unit-scaling`, run:

.. code-block::

    pip install git+https://github.com/graphcore-research/unit-scaling.git

Getting Started
---------------

We recommend that new users get started with :numref:`User Guide`.

A reference outlining our API can be found at :numref:`API reference`.

The following video gives a broad overview of the workings of unit scaling.

.. raw:: html
   :file: _static/animation.html

.. Note:: The library is currently in its *beta* release.
    Some features have yet to be implemented and occasional bugs may be present.
    We're keen to help users with any problems they encounter.

Development
-----------

For those who wish to develop on the :code:`unit-scaling` codebase, clone or fork our
`GitHub repo <https://github.com/graphcore-research/unit-scaling.git>`_ and follow the
instructions in our :doc:`developer guide <development>`.

.. toctree::
    :caption: Contents
    :numbered:
    :maxdepth: 3

    User guide <user_guide>
    Developer guide <development>
    Limitations <limitations>
    Blog <blog>
    API reference <api_reference>
