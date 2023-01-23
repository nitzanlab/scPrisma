scPrisma
==============================================
.. toctree::
    :caption: General
    :maxdepth: 4
    :hidden:

    scPrisma
    Installation
    Cyclic signals

.. image:: https://github.com/nitzanlab/scPrisma/blob/master/workflow.png?raw=true
   :width: 600px
   :align: center

**scPrisma** scPrisma is a spectral analysis method, for pseudotime reconstruction, informative genes inference, filtering, and enhancement of underlying cyclic signals

Installation and pre-processing
-------

pip
~~~~~~~~~~
scPrisma can be installed using pip (cpu version)::

    git clone https://github.com/nitzanlab/scPrisma.git
    cd scPrisma
    pip install .

GPU (pytorch) version::

    git clone https://github.com/nitzanlab/scPrisma.git
    cd scPrisma
    pip install ."[gpu]"


scPrisma is based on matrix optimization using gradient descent, therefore using GPU boosts the performance significantly. It is highly recommended using GPU for large datasets. e.g for the reconstruction task it is recommended using GPU for datasets which are larger than 2,000-3,000 cells, and for the gene inference/filtering/enhancement for datasets which are larger than 15,000-20,000 cells.

import scanpy as sc

Imports and pre-processing
~~~~~~~~~~
We highly recommend using `scanpy <https://scanpy.readthedocs.io/>`_ for preprocessing, visualization, and downstream analysis of scRNA-seq data.
First, import 'scPrisma', 'scanpy' and 'numpy'. If you use the GPU version, import also 'torch'::

    import scanpy as sc
    import numpy as np
    import scPrisma
    #import torch



Cyclic signals Workflows
-------
The core of scPrisma utilizes spectral template matching between the spectrum (the eigendecomposition of the covariance matrix) of a set of single-cell data (e.g. scRNA-seq), and the expected analytical spectrum of a structure or process we aim to enhance or filter.

1. The enhancement workflow consists of three stages: reconstruction of the cyclic signal (order the cells along the topology), infer genes which are smooth over the topology (gene inference), from the inferred genes retain only expression profiles which are related to the inferred signal.

2. The filtering workflow consist of two stages: reconstruction of the cyclic signal, filter out expression profiles which are not related to the inferred signal.

Reconstruction
~~~~~~~~~~
The first step in each one of the workflows is reconstructing the signal (order the cells along the topology). 
This can be done in few ways:

1. Using the reconstruction algorithm.

2. Using full prior knowledge 

2. The filtering workflow consist of two stages: reconstruction of the cyclic signal, filter out expression profiles which are not related to the inferred signal.


this can be done using the reconstriction algorithm:


General topology
-------


Reconstruction
~~~~~~~~~~
