#!/bin/zsh
#SBATCH --job-name=fp-objectnav-ver-v0.1.1
#SBATCH --output=data/training/floorplanner/ver_v0.1.1/slurm.out
#SBATCH --error=data/training/floorplanner/ver_v0.1.1/slurm.err
#SBATCH --gpus 8
#SBATCH --nodes 1
#SBATCH --cpus-per-task 16
#SBATCH --ntasks-per-node 8
#SBATCH --partition=short
#SBATCH --constraint=a40
#SBATCH --exclude=nestor,roberto,deebot

MAIN_ADDR=$(srun --ntasks=1 hostname 2>&1 | tail -n1)
export MAIN_ADDR
export CUDA_LAUNCH_BLOCKING=1
echo $SLURM_JOB_ID
source ~/.zshrc
conda activate sbd
cd /nethome/mkhanna37/flash1/sbd/fphab/habitat-lab

export GLOG_minloglevel=2
export MAGNUM_LOG=quiet
export HABITAT_SIM_LOG=quiet
export PYTHONPATH=$PYTHONPATH:./

srun python -u habitat-baselines/habitat_baselines/run.py --exp-config habitat-baselines/habitat_baselines/config/objectnav/ver_objectnav_fp.yaml --run-type train
