Installation
==============================================
.. toctree::
    :caption: General
    :maxdepth: 4
    :hidden:

    scPrisma
    Installation
    Cyclic signals

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



