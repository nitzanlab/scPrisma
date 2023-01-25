# scPrisma
scPrisma is a spectral analysis method, for pseudotime reconstruction, informative genes inference, filtering, and enhancement of underlying topological signals.
![workflow](https://github.com/nitzanlab/scPrisma/blob/master/workflow.png?raw=true)

<!-- Manuscript -->
## Manuscript
[scPrisma manuscript](https://www.biorxiv.org/content/10.1101/2022.06.07.493867v1)

<!-- GETTING STARTED -->
## Getting Started
For documentation please refer to [scPrisma documentation](https://scprisma.readthedocs.io/en/latest/).

<!-- Reproducibility -->
## Reproducibility
<h4> For reproducibility of scPrisma manuscript, please refer to:<br /> https://github.com/nitzanlab/scPrisma_notebooks</h4>

<!-- Installation -->
## Installation

```sh
git clone https://github.com/nitzanlab/scPrisma.git
cd scPrisma
pip install .
For running the gpu version install it like so pip install ."[gpu]"
```
<br />

<!-- Tutorials -->
## Tutorials

[CPU tutorials](https://github.com/nitzanlab/scPrisma/tree/master/tutorials/cpu)
[GPU tutorials](https://github.com/nitzanlab/scPrisma/tree/master/tutorials/gpu)


<!-- TABLE OF CONTENTS -->
## Table of Contents

* [About the Project](#about-the-project)
  * [Built With](#built-with)
* [Getting Started](#getting-started)
  * [Reconstruction](#reconstruction)
  * [Filtering workflow](#filtering-workflow)
  * [Enhancement workflow](#enhancement-workflow)
* [Tutorials](#tutorials)
* [Contact](#contact)



<!-- ABOUT THE PROJECT -->
## About The Project
[Manuscript](https://www.biorxiv.org/content/10.1101/2022.06.07.493867v1) <br />
[Notebooks](https://github.com/nitzanlab/scPrisma_notebooks)
### Built With
* [Python](https://www.python.org/) - 3.7 , Numpy and Numba libraries. it is recommended to use also Scanpy library.



<!-- GETTING STARTED -->
## Getting Started

```sh
git clone https://github.com/nitzanlab/scPrisma.git
cd scPrisma
pip install .
For running the gpu version install it like so pip install ."[gpu]"
```

##
## Imports
It is recommended to use ['scanpy'](https://scanpy.readthedocs.io/en/stable/index.html) package. 

```
from scPrisma.algorithms import *
import scanpy as sc
import numpy as np
```
## Pre-processing
```
adata = sc.read(path)
sc.pp.normalize_per_cell(adata, counts_per_cell_after=1e4)
sc.pp.log1p(adata)
```
'adata' should be AnnData object, where the observations (rows) are cells and the variables (columns) are genes. 
## Reconstruction
The first step in this workflow is reconstruct the signal, this can be done using the reconstriction algorithm:

```
E , E_rec = reconstruction_cyclic(adata.X)
order = E_to_range(E_rec)
adata = adata[order,:]
```
'reconstruction_cyclic' function receives as an input the gene expression matrix (and parameters that can be seen in 'algorithms.py') and returns 'E' which is a doubly stochastic matrix and 'E_rec' which is the desired permutation matrix.
'E_to_range' turns the permutation matrix into a permutation array.

If low-resolution pseudotime ordering exists (as prior knowledge) it can be used instead of applying the reconstruction algorithm:
```
adata = sort_data_crit(adata=adata ,crit=crit,crit_list=crit_list)
```

Each cell should have a label of his place stored as 'obs'. 'crit' is the desired label,  'crit_list' is the desired order.
For example: for sorting a gene expression that was sampled at four different timepoints (0,6,12,18). The sampling time can be stored as 'ZT' (adata.obs['ZT'] = X). and applied the following function:
```
adata = sort_data_crit(adata=adata ,crit='ZT',crit_list=['0','6','12','18'])
```
The best performance would be achieved if there were similar numbers of samples in each state. Subsampling states with more cells than others can solve this issue. 



## Filtering workflow
After reconstruction was applied, we can use the filtering algorithm. This algorithm filters out the expression profiles that are related to the reconstructed topology.
```
F = filtering_cyclic(adata.X, regu=0 )
adata.X = adata.X * F
```
'regu' is the regularization parameter, it is recomended that this parameter would be between 0 and 0.5. As long as we increase this parameter <b><u>less</u></b> information would be filter out. Since it is a convex optimization problem, it is solved using backtracking line search gradient descent.
## Enhancement workflow
After reconstruction was applied, we can use the enhancement algorithm. This algorithm filters out the expression profiles that <b><u>are not</u></b> related to the reconstructed topology.
It is recomended to use the informative genes infereence algorithm before using the enhancement algorithm. Running the genes inference algorithm, prevents overfitting of genes that are not related to desired topology.
```
D = filter_cyclic_genes_line(adata.X, regu=0)
D = np.identify(D.shape[0])-D
adata.X = (adata.X).dot(D)
```
'regu' is the regularization parameter, it is recomended that this parameter would be between -0.1 and 0.5. As long as we increase this parameter the algorithm would filter out <b><u>less</u></b> genes. But, we will retain the genes that the algorithm would not filter, so as long as we increase this parameter,<b><u>more</u></b> genes will be filtered out.
Next we can apply the enhancement algorithm:

```
F = enhancement_cyclic(adata.X, regu=0 )
adata.X = adata.X * F
```

As long as we increase the regularization parameter we will filter out <b><u>more</u></b> information.


<!-- TUTORIALS -->
## Tutorials
[De-novo reonstruction, cyclic enhancement and filtering- HeLa S3 cells](https://github.com/nitzanlab/scPrisma/blob/main/tutorials/tutorial_de_novo_reconstruction.ipynb)
<br />
[Reonstruction from prior knowledge, cyclic enhancement and filtering, linear enhancment and filtering- Mouse liver lobules](https://github.com/nitzanlab/scPrisma/blob/main/tutorials/tutorial_prior_knowledge_linear_and_cyclic.ipynb)

## Running the tests

I recommend creating two separate virtual environments for running the cpu/gpu test suite. On my laptop, I use conda but this can be replaced any other virtual environment manager of your choice.

### Running the cpu only tests 

```
conda create -n scprisma_cpu python=3.10
conda activate scprisma_cpu
pip install .
pytest tests/cpu
```

### Running the gpu only tests 

```
conda create -n scprisma_gpu python=3.10
conda activate scprisma_gpu
pip install .[gpu]
pytest tests/gpu
```

<!-- CONTACT -->
## Contact
Jonathan Karin - jonathan.karin [at ] mail.huji.ac.il
