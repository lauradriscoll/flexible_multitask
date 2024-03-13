#!/bin/bash
#
#SBATCH --job-name=512_l2_w6_h7
#
#SBATCH --time=15:00:00
#!/bin/bash
#SBATCH -p gpu
#SBATCH -c 10
#SBATCH --gres gpu:1

ml python/2.7.13
module load py-scipystack/1.0_py27 py-tensorflow/1.9.0_py27
srun python general_model_train.py --gres
