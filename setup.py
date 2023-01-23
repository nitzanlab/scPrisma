from setuptools import setup

requirements = [
    "numpy",
    "scanpy",
    "numba==0.55.2",
    "pandas",
    "scikit-learn",
    "scipy",
    "seaborn",
    "matplotlib",
    "pytest", #TODO: For simplicity adding it to requirements.
]
gpu = ["torch"]

setup(
    name="scPrisma",
    version="0.0.5",
    packages=["scPrisma"],
    url="https://github.com/nitzanlab/scPrisma",
    license="MIT License",
    author="jonathankarin",
    author_email="jonathan.karin@mail.huji.ac.il",
    maintainer="Haimasree Bhattacharya",
    maintainer_email="haimasree.il@gmail.com",
    description="scPrisma: inference, filtering and enhancement of periodic signals in single-cell data using spectral template matching ",
    python_requires=">=3",
    install_requires=requirements,
    extras_require={"gpu": gpu},
)
