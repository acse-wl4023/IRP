#!/bin/bash -l

#SBATCH --job-name=ner
#SBATCH --partition=dgxl_irp
#SBATCH --qos=dgxl_irp_high

####--cpus-per-task 8 --gres=gpu:1 --mem-per-cpu 1500 --time 24:00:00 --pty bash -i
#SBATCH -e /scratch_dgxl/sc4623/wl4023/slurm-%j.err              # File to redirect stderr
#SBATCH -o /scratch_dgxl/sc4623/wl4023/slurm-%j.out              # File to redirect stdout
#SBATCH --mem=16384M                   # Memory per processor
#SBATCH --time=24:00:00              # The walltime
#SBATCH --nodes=1                    # Run all processes on a single node
#SBATCH --ntasks=1                   # Number of tasks
##SBATCH --ntasks-per-socket=1       # Maximum number of tasks on each socket
#SBATCH --cpus-per-task=1            # CPUs per task
#SBATCH --gres=gpu:1                 # Number of GPUs


source /scratch_dgxl/sc4623/miniconda3/etc/profile.d/conda.sh
conda activate wl


python /scratch_dgxl/sc4623/wl4023/IRP/BaselineModels/GConvLSTM/train.py
