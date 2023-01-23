scPrisma
==============================================
.. toctree::
    :caption: General
    :maxdepth: 4
    :hidden:

    scPrisma
    Installation

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
First, import ``scPrisma``, ``scanpy`` and ``numpy``. If you use the GPU version, import also ``torch``::

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

1. Using the reconstruction algorithm, if there is no any prior knowledge and the signal is strong enough (as described in the manuscript). Usage example can be found in this `tutorial <https://github.com/nitzanlab/scPrisma/blob/master/tutorials/tutorial_de_novo_reconstruction.ipynb>`_. If you use the GPU version, you should use ``scPrisma.algorithms_torch.reconstruction_cyclic_torch`` and ``scPrisma.algorithms_torch.e_to_range_torch`` instead of the mentioned functions::

      E , E_rec = scPrisma.algorithms.reconstruction_cyclic(adata.X)
      order = scPrisma.algorithms.e_to_range(E_rec)
      adata = adata[order,:]

2. Using full prior knowledge. Use the function ``sort_data_crit`` which sorts the according to an observation of the field that is inserted as 'crit', in the order of 'crit_list' (using scanpy AnnData). In the example, the observation is 'ZT' which contains the values: '0','6','12','18'. After the sorting, the cells with the values '0' would be the first in the expression matrix, than '6','12' and '18'::

      scPrisma.algorithms_torch.sort_data_crit(adata, crit='ZT',crit_list=['0','6','12','18'])


3. Using partial prior knowledge:
A. partial labels.



he filtering workflow consist of two stages: reconstruction of the cyclic signal, filter out expression profiles which are not related to the inferred signal.


this can be done using the reconstriction algorithm:


General topology
-------


Reconstruction
~~~~~~~~~~
