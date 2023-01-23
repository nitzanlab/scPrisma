Cyclic signals
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

Cyclic signals Workflows
-------
The core of scPrisma utilizes spectral template matching between the spectrum (the eigendecomposition of the covariance matrix) of a set of single-cell data (e.g. scRNA-seq), and the expected analytical spectrum of a structure or process we aim to enhance or filter.

1. The enhancement workflow consists of three stages: reconstruction of the cyclic signal (order the cells along the topology), infer genes which are smooth over the topology (gene inference), from the inferred genes retain only expression profiles which are related to the inferred signal.

2. The filtering workflow consist of two stages: reconstruction of the cyclic signal, filter out expression profiles which are not related to the inferred signal.

Reconstruction
~~~~~~~~~~
The first step in each one of the workflows is reconstructing the signal (order the cells along the topology). 
This can be done in few ways:

1. Using the reconstruction algorithm, if there is no any prior knowledge and the signal is strong enough (as described in the manuscript)::

      E , E_rec = scPrisma.algorithms.reconstruction_cyclic(adata.X)
      order = scPrisma.algorithms.e_to_range(E_rec)
      adata = adata[order,:]


2. Using full prior knowledge 

2. The filtering workflow consist of two stages: reconstruction of the cyclic signal, filter out expression profiles which are not related to the inferred signal.


this can be done using the reconstriction algorithm:

