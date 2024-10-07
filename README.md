# SLURM Batch Script for Running Python on GPU

# If per-batch training loss is required to be printed, mention --print-per-batch-loss
# For different number of workers we can change the number of workers in the last line, for e.g., for 0 workers: --num-workers 0
# to run on CPU, remove --CUDA. This is because by default, if code does not specify a device for computation (e.g., GPU or CPU), it will run on the CPU. 
# For different optimizers we can change the optimizer name in the last line, for e.g., for SGD with Nesterov we can write: --optimizer sgd_nesterov
# for question C7, we need to change the input file name to python c7.py (in the last line)
## SLURM Batch Script

```bash
#!/bin/bash

#SBATCH --account=ece_gy_9143-2024fa
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu
#SBATCH --job-name=c2
#SBATCH --nodes=1
#SBATCH --output=%x.out
#SBATCH --time=02:00:00
#SBATCH --requeue

cd /scratch/pm3667/lab2

singularity exec --nv --overlay my_pytorch.ext3:ro /scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
    /bin/bash -c "source /ext3/env.sh; python c1.py --CUDA --num-worker 4 --optimizer SGD --print-per-batch-loss"


