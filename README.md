# scPrisma
scPrisma is a workflow based on projection over theoretic covariance spectrum for pseudotime reconstruction, informative genes inference and signals filtering and enhancement. 

<br />


<!-- TABLE OF CONTENTS -->
## Table of Contents

* [About the Project](#about-the-project)
  * [Built With](#built-with)
* [Getting Started](#getting-started)
  * [Reconstruction](#reconstruction)
  * [Filtering workflow](#filtering-workflow)
  * [Enhancement workflow](#enhancement-workflow)
* [Contact](#contact)



<!-- ABOUT THE PROJECT -->
## About The Project
---

### Built With
* [Python](https://www.python.org/) - 3.7 , Numpy and pandas libraries. 



<!-- GETTING STARTED -->
## Getting Started

```sh
git clone https://github.com/nitzanlab/scPrisma.git
cd scPrisma
pip install .
```
## Pre-processing
It is recommended to use ['scanpy'](https://scanpy.readthedocs.io/en/stable/index.html) package. 
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


## Filtering workflow
For filtering we need to first 
```sh
python ....
```
## Enhancement workflow
```sh
python ...
```
....


<!-- CONTACT -->
## Contact
....
