# scPrisma
scPrisma is a workflow based on projection over theoretic covariance spectrum for pseudotime reconstruction, informative genes inference and signals filtering and enhancement. 

<br />


<!-- TABLE OF CONTENTS -->
## Table of Contents

* [About the Project](#about-the-project)
  * [Built With](#built-with)
* [Getting Started](#getting-started)
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
## Reconstruction
Before 

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
