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

Trained networks were deposited on the Allen Institute database 'multitask_shared', and additional processed data to generate all figures, including fixed point locations, were deposited at 'multitask_processed' : https://nam12.safelinks.protection.outlook.com/?url=https%3A%2F%2Fopen.quiltdata.com%2Fb%2Faind-trained-networks%2Ftree%2F&data=05%7C02%7C%7C304a702ec6e34e8c06b108dce85f3cca%7C32669cd6737f4b398bddd6951120d3fc%7C0%7C0%7C638640742977336536%7CUnknown%7CTWFpbGZsb3d8eyJWIjoiMC4wLjAwMDAiLCJQIjoiV2luMzIiLCJBTiI6Ik1haWwiLCJXVCI6Mn0%3D%7C0%7C%7C%7C&sdata=FtBs%2BWI3QWA6rNhy0iUJgGpwTlnE9oI2FGMhbYACdMw%3D&reserved=0

## Citation

If you use this code in your research, please cite the following paper:

Driscoll, L.N., Shenoy, K. & Sussillo, D. Flexible multitask computation in recurrent networks utilizes shared dynamical motifs. Nat Neurosci 27, 1349–1363 (2024). https://doi.org/10.1038/s41593-024-01668-6
