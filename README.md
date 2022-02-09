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
After reconstruction was applied, we can use the filtering algorithm. This algorithm filters out the expression profiles that are related to the reconstructed topology.
```
F = filtering_cyclic(adata.X, regu=0 )
adata.X = adata.X * F
```
'regu' is the regularization parameter, it is recomended that this parameter would be between 0 and 0.5. As long as we increase this parameter <b><u>less</u></b> information would be filter out. Since it is a convex optimization problem, it is solved using backtracking line search gradient descent.
## Enhancement workflow
After reconstruction was applied, we can use the enhancement algorithm. This algorithm filters out the expression profiles that <b><u>are not</u></b> related to the reconstructed topology.
It is recomended to use the informative genes infereence algorithm before using the enhancement algorithm. Running the genes inference algorithm, prevents overfitting of genes that do not related to desired topology.
```
D
```
'regu' is the regularization parameter, it is recomended that this parameter would be between 0 and 0.5. As long as we increase this parameter <b><u>less</u></b> information would be filter out.

```sh
python ...
```
....


<!-- CONTACT -->
## Contact
....
