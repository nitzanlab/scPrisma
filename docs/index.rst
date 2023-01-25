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
The first step in each one of the workflows is reconstructing the signal (order the cells along the topology). This stage is the most challenging stage, both in terms of computational complexity and accuracy.

This can be done in few ways:

1. Using the reconstruction algorithm, if there is no any prior knowledge and the signal is strong enough (as described in the manuscript). Usage example can be found in this `tutorial <https://github.com/nitzanlab/scPrisma/blob/master/tutorials/tutorial_de_novo_reconstruction.ipynb>`_. If you use the GPU version, you should use ``scPrisma.algorithms_torch.reconstruction_cyclic_torch`` and ``scPrisma.algorithms_torch.e_to_range_torch`` instead of the mentioned functions::

      E , E_rec = scPrisma.algorithms.reconstruction_cyclic(adata.X)
      order = scPrisma.algorithms.e_to_range(E_rec)
      adata = adata[order,:]

2. Using full prior knowledge. Use the function ``sort_data_crit`` which sorts the according to an observation of the field that is inserted as 'crit', in the order of 'crit_list' (using scanpy AnnData). In the example, the observation is 'ZT' which contains the values: '0','6','12','18'. After the sorting, the cells with the values '0' would be the first in the expression matrix, than '6','12' and '18'::

      scPrisma.algorithms_torch.sort_data_crit(adata, crit='ZT',crit_list=['0','6','12','18'])


3. Using partial prior knowledge- discrete labels restricting the optimization parameter (E) to a specific subset of doubly stochastic matrices by creating an indicator matrix. After each gradient ascent step, the optimization parameter is multiplied by this indicator matrix. An example is available in the following  
`tutorial <https://github.com/nitzanlab/scPrisma/blob/master/tutorials/hematopoietic_progenitors_reconstruction_with_partial_prior_knowledge.ipynb>`_.
A. For example, given cell cycle phase labels ('G1', 'S', 'G2M') we will divide the optimization parameter to three consecutive bins, the first will be for cells of 'G1' phase, the second for cells of 'S' phase and the third of 'G2M' phases::

      indicator_matrix = np.zeros((adata_reconstruction.n_obs,adata_reconstruction.n_obs))
      crit_list = ['G1', 'G2M', 'S']
      for i , j in enumerate(adata.obs["phase"]):
            indicator_matrix[i,:]=np.array(adata.obs["phase"]==j,dtype=int)
      E, E_rec = scPrisma.algorithms.reorder_indicator(adata.X,indicator_matrix, iterNum=100)
      order = scPrisma.algorithms.e_to_range(E_rec)
      adata = adata[order,:]


4. Using partial prior knowledge- gene selection, the reconstruction algorithm can be applied to only a subset of genes that are known to be related to the desired signal. It is important to ensure that the selected subset of genes contains information about all the different states of the desired signal. For example, using only marker genes for the 'S' and 'G2M' phases of the cell cycle for reconstruction, would not produce accurate results. In such cases, it may be more beneficial to increase the proportion of expression of those marker genes within the total expression, for example, by multiplying each one by a constant greater than 1.



he filtering workflow consist of two stages: reconstruction of the cyclic signal, filter out expression profiles which are not related to the inferred signal.


this can be done using the reconstriction algorithm:

Enhancement workflow
~~~~~~~~~~

Genes inference
******
After obtaining the reconstructed signal, either through solving the reconstruction problem or by using prior knowledge, we will proceed to identify the genes that are relevant to the desired signal.

This can be done in two ways:

1. Using the genes inference algorithm. Due to convexity considerations, it is easier to infer genes that are not related to the desired signal, and then flip the results. The regularization parameter is the only parameter that should be tuned. By keeping a large value, the  algorithm would keep more cyclic genes, which would then be filtered later by flipping. The optimal way to tune this parameter is when we have known 'flat' genes (which we want to filter out) and known 'cyclic' genes (which we want to keep). An example for the tuning of this parameter is available in this `tutorial <https://github.com/nitzanlab/scPrisma/blob/master/tutorials/tutorial_de_novo_reconstruction.ipynb>`_. If you use the GPU version, you should use ``scPrisma.algorithms_torch.filter_cyclic_genes_torch`` instead of the mentioned functions::

      adata_enhancement= adata.copy()
      D = scPrisma.algorithms.filter_cyclic_genes(adata_enhancement.X,regu=0, iterNum=100)
      adata_enhancement.X = adata_enhancement.X @ D

2. Another possibility is to select a fixed number of genes to retain based on their projection over the theoretical spectrum. This is similar to adjusting the regularization parameter to retain this number of genes.  An example is available in the following  
`tutorial <https://github.com/nitzanlab/scPrisma/blob/master/tutorials/hematopoietic_progenitors_reconstruction_with_partial_prior_knowledge.ipynb>`_. If you use the GPU version, you should use ``scPrisma.algorithms_torch.filter_non_cyclic_genes_by_proj_torch`` instead of the mentioned functions::

      adata_enhancement= adata.copy()
      D = scPrisma.algorithms.filter_non_cyclic_genes_by_proj(adata_enhancement.X,n_genes=500)
      adata_enhancement.X = adata_enhancement.X @ D


Enhancement
******
Next, We will clear from them the expression profiles which are not related to the desired signal.
Here we also need to tune the regularization parameter. Higher regularization will filter out more expression profiles which are not related to the reconstructed signal. An example is available in the following  
`tutorial <https://github.com/nitzanlab/scPrisma/blob/master/tutorials/hematopoietic_progenitors_reconstruction_with_partial_prior_knowledge.ipynb>`_. If you use the GPU version, you should use ``scPrisma.algorithms_torch.enhancement_cyclic_torch`` instead of the mentioned functions::

      F = scPrisma.algorithms.enhancement_cyclic(adata_enhancement.X,regu=0.01)
      adata_enhancement.X = adata_enhancement.X * F

Filtering workflow
~~~~~~~~~~
If insead of enhancing the reconstructed signal, we want to filter it out we can use cyclic filtering algorithm. Here, we also have a regularization parameter, higher regularization will keep more expression profiles. An example is available in the following  
`tutorial <https://github.com/nitzanlab/scPrisma/blob/master/tutorials/hematopoietic_progenitors_reconstruction_with_partial_prior_knowledge.ipynb>`_. If you use the GPU version, you should use ``scPrisma.algorithms_torch.adata_filtering`` instead of the mentioned functions::

      adata_filtering= adata.copy()
      F = scPrisma.algorithms.filtering_cyclic(adata_filtering.X,regu=0.01)
      adata_filtering.X = adata_filtering.X * F


General topology
-------
In addition to the circular topology, it is possible to enhance/filter out a topology which is represented by a covaraince matrix designed by the user.

Reconstruction
~~~~~~~~~~
