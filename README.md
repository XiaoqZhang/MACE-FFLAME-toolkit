# FFLAME: A Fragment-to-Framework Learning Approach for MOF Potentials

A toolkit for fine-tuning MACE

## Functions

* Extract ligands from MOFs and add hydrogen atoms and optimize the ligands. 

* Run molecular dynamics simulations with various ase calculators. 

* Select configurations for training using KMeans clustering method. 

* Fine-tune MACE.


## Installation

We recommend installing fflame in a clean conda environment. Follow these steps:

1. Create and activate the environment
```
conda create -n fflame python=3.10 -y
conda activate fflame
```

2. Install [PyTorch](https://pytorch.org/get-started/locally/) and [MACE](https://github.com/ACEsuit/mace)

3. Install dependencies and fflame
```
conda install xtb-python -c conda-forge
pip install .
# install mofchecker
pip install git+https://github.com/Au-4/mofchecker_2.0
```

## How to use

The example scripts are put in `experiments`.

## References

If you use this code, please cite our paper:

