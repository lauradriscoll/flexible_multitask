#!/bin/bash
#
#SBATCH --job-name=fp_finding_stepX
#
#SBATCH --time=10:00:00
#!/bin/bash
#SBATCH -p gpu
#SBATCH -c 1
#SBATCH --gres gpu:1

ml python/2.7.13
module load py-scipystack/1.0_py27 py-tensorflow/1.9.0_py27
srun python find_fp_stepX_tasks_transfer.py --gres
