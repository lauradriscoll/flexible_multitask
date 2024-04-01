# Flexible Multitask Computation in Recurrent Networks

## Overview

This repository contains the code associated with the paper titled "Flexible Multitask Computation in Recurrent Networks Utilizes Shared Dynamical Motifs". The code implements training the recurrent neural network models and the analyses described in the paper. https://www.biorxiv.org/content/10.1101/2022.08.15.503870v1

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
pip install -e .
```

## Usage

- `stepnet`: Code for training networks with random initializations.

- `transfer_learn`: Code for training networks starting from a pretrained network.

- `utils`: This folder contains utility functions used across different parts of the codebase, such as data preprocessing, etc.

- `analysis`: Reproduces figures in paper using open source data.

## Data

The data used in the experiments are not included in this repository due to size constraints. However, data will be made available through the Allen Institute.

## Citation

If you use this code in your research, please cite the following paper:

Flexible multitask computation in recurrent networks utilizes shared dynamical motifs
Laura Driscoll, Krishna Shenoy, David Sussillo
bioRxiv 2022.08.15.503870; doi: https://doi.org/10.1101/2022.08.15.503870
