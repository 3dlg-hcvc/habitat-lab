#!/bin/zsh
#SBATCH --job-name=fp-objectnav-ddppo-v0.2.0_6_cat_indoor_only
#SBATCH --output=data/training/floorplanner/ddppo/v0.2.0_6_cat_indoor_only/slurm.out
#SBATCH --error=data/training/floorplanner/ddppo/v0.2.0_6_cat_indoor_only/slurm.err
#SBATCH --gpus 8
#SBATCH --nodes 1
#SBATCH --cpus-per-task 16
#SBATCH --ntasks-per-node 8
#SBATCH --partition=short
#SBATCH --constraint=a40

MAIN_ADDR=$(srun --ntasks=1 hostname 2>&1 | tail -n1)
export MAIN_ADDR
export CUDA_LAUNCH_BLOCKING=1
echo $SLURM_JOB_ID
source ~/.zshrc
conda activate sbd
cd /nethome/mkhanna37/flash1/sbd-latest/scene-builder-datasets/fphab/habitat-lab

export GLOG_minloglevel=2
export MAGNUM_LOG=quiet
export HABITAT_SIM_LOG=quiet
export PYTHONPATH=$PYTHONPATH:./

srun python -u habitat-baselines/habitat_baselines/run.py --exp-config habitat-baselines/habitat_baselines/config/objectnav/ddppo/ddppo_objectnav_fp_6_cat.yaml --run-type train
