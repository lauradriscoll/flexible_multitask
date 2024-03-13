# Flexible Multitask Computation in Recurrent Networks

## Overview

This repository contains the code associated with the paper titled "Flexible Multitask Computation in Recurrent Networks Utilizes Shared Dynamical Motifs". The code implements training the recurrent neural network models and the analyses described in the paper. https://www.biorxiv.org/content/10.1101/2022.08.15.503870v1

## Requirements

- Python (2.7.5)
- TensorFlow (1.10.0)
- Other dependencies as specified in the requirements.txt file

## Installation

1. Clone this repository to your local machine:

```
git clone https://github.com/lauradriscoll/flexible_multitask.git
```

2. Install the required dependencies:

```
pip install -r requirements.txt
```

## Usage

- `train_model.py`: This script trains the recurrent neural network models on the specified multitask learning problem. You can specify various parameters such as network architecture, learning rate, batch size, etc., through command-line arguments.

- `evaluate_model.py`: After training the models using `train_model.py`, you can use this script to evaluate the performance of the trained models on held-out test data. It computes relevant metrics and generates visualizations for analysis.

- `utils.py`: This script contains utility functions used across different parts of the codebase, such as data loading, preprocessing, etc.

- analysis folder: Reproduces figures in paper using data from, XXX

## Data

The data used in the experiments are not included in this repository due to size constraints. However, instructions for obtaining the data and preparing it for use with the code can be found in the `data/README.md` file.

## Citation

If you use this code in your research, please cite the following paper:

Flexible multitask computation in recurrent networks utilizes shared dynamical motifs
Laura Driscoll, Krishna Shenoy, David Sussillo
bioRxiv 2022.08.15.503870; doi: https://doi.org/10.1101/2022.08.15.503870
