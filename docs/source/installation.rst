Installation Instructions
=========================

Installation
------------
``pip install kdescent``

Prerequisites
-------------
- Tested on Python versions: ``python=3.9-12``
- JAX (GPU install available - see https://jax.readthedocs.io/en/latest/installation.html)

Example installation with conda env:
++++++++++++++++++++++++++++++++++++

.. code-block:: bash

    conda create -n py312 python=3.12
    conda activate py312
    conda install -c conda-forge jax
    pip install kdescent
