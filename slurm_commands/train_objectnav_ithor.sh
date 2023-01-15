#!/bin/sh
#SBATCH --account=def-msavva
#SBATCH --nodes=1           # total nodes
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:a100:4        # how many GPUs per node
#SBATCH --cpus-per-task=12   # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --mem=0          # Memory proportional to GPUs: 32000 Cedar, 64000 Graham.
#SBATCH --time=2-23:00
#SBATCH --job-name=objnav_ithor
#SBATCH --output=/home/hanxiao/slurm_outs/%N-%j.out
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=shawn_jiang@sfu.ca


module load python/3.7.9
module load cuda/11.7

source /home/hanxiao/habfp/bin/activate
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

srun --ntasks=$SLURM_NNODES --ntasks-per-node=1 bash slurm_commands/init_setup.sh

wait

MAIN_ADDR=$(srun --ntasks=1 hostname 2>&1 | tail -n1)
export MAIN_ADDR

export GLOG_minloglevel=2
export MAGNUM_LOG="quiet"
export HABITAT_SIM_LOG=quiet

set -x
srun python -u habitat-baselines/habitat_baselines/run.py --exp-config habitat-baselines/habitat_baselines/config/objectnav/ddppo_objectnav_ithor.yaml --run-type train
