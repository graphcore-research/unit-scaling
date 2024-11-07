Unit Scaling
============

Welcome to the :code:`unit-scaling` library. This library is designed to facilitate
the use of the *unit scaling* and *u-µP* methods, as outlined in the papers
`Unit Scaling: Out-of-the-Box Low-Precision Training (ICML, 2023)
<https://arxiv.org/abs/2303.11257>`_ and
`u-μP: The Unit-Scaled Maximal Update Parametrization
<https://arxiv.org/abs/2407.17465>`_

For a demonstration of the library, see `u-μP using the unit_scaling library
<https://github.com/graphcore-research/unit-scaling/blob/main/examples/demo.ipynb>`_ — a notebook showing the definition and training of a u-µP language model, comparing against Standard Parametrization (SP).

Installation
------------

To install :code:`unit-scaling`, run:

.. code-block::

    pip install unit-scaling

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

`The following slides <https://github.com/graphcore-research/unit-scaling/blob/main/docs/u-muP_slides.pdf>`_ also give an overview of u-µP.

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
    API reference <api_reference>
