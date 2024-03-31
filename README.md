# scPrisma
scPrisma is a spectral analysis method, for pseudotime reconstruction, informative genes inference, filtering, and enhancement of underlying topological signals.
![workflow](https://github.com/nitzanlab/scPrisma/blob/master/workflow.png?raw=true)

<!-- Manuscript -->
## Manuscript
[Karin, J., Bornfeld, Y. & Nitzan, M. scPrisma infers, filters and enhances topological signals in single-cell data using spectral template matching. Nat Biotechnol (2023)](https://www.nature.com/articles/s41587-023-01663-5)

<!-- GETTING STARTED -->
## Getting Started
For documentation please refer to [scPrisma documentation](https://scprisma.readthedocs.io/en/latest/).

<!-- Reproducibility -->
## Tutorials
<h4> For tutorials, please refer to:<br /> [Tutorials](https://github.com/nitzanlab/scPrisma/tree/master/tutorials)</h4>

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
### To cite:
```
@article{karin2023scprisma,
  title={sc{P}risma infers, filters and enhances topological signals in single-cell data using spectral template matching},
  author={Karin, Jonathan and Bornfeld, Yonathan and Nitzan, Mor},
  journal={Nature Biotechnology},
  pages={1--10},
  year={2023},
  publisher={Nature Publishing Group US New York}
}
```


<!-- CONTACT -->
## Contact
Jonathan Karin - jonathan.karin [at ] mail.huji.ac.il <br />
[Forum](https://github.com/nitzanlab/scPrisma/discussions)
