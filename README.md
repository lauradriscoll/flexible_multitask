# Flexible Multitask Computation in Recurrent Networks

## Overview

This repository contains the code associated with the paper titled "Flexible multitask computation in recurrent networks utilizes shared dynamical motifs". The code implements training the recurrent neural network models and the analyses described in the paper. https://www.nature.com/articles/s41593-024-01668-6

## Requirements

- Python (3.10)
- TensorFlow (2)
- Other dependencies as specified in the requirements.txt file

## Installation

1. Clone this repository to your local machine:

```
git clone https://github.com/lauradriscoll/flexible_multitask.git
```

2. Install the required dependencies:

```
conda create --name flex_mult python=3.10
conda activate flex_mult
cd flexible_multitask
vim .env [download processed data from links below and update path to data and this repo]
pip install -e .
```

## Usage

- `stepnet`: Code for training networks with random initializations.

- `transfer_learn`: Code for training networks starting from a pretrained network.

- `utils`: This folder contains utility functions used across different parts of the codebase, such as data preprocessing, etc.

- `analysis`: Reproduces figures in paper using open source data.

## Data

Trained networks were deposited on the Allen Institute database at https://codeocean.allenneuraldynamics.org/data-assets/e44a8ae2-255a-40f1-ac77-e7d04975ac8b, and additional processed data to generate all figures, including fixed point locations, were deposited at https://codeocean.allenneuraldynamics.org/data-assets/c4f26da4-73b5-4fb6-a9f2-cba52e3cf400/multitask_processed/.

## Citation

If you use this code in your research, please cite the following paper:

Driscoll, L.N., Shenoy, K. & Sussillo, D. Flexible multitask computation in recurrent networks utilizes shared dynamical motifs. Nat Neurosci 27, 1349–1363 (2024). https://doi.org/10.1038/s41593-024-01668-6
