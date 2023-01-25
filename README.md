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

<!-- Tests -->

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
Jonathan Karin - jonathan.karin [at ] mail.huji.ac.il <br />
[Forum](https://github.com/nitzanlab/scPrisma/discussions)
