#!/bin/bash
#SBATCH --job-name=DPSGD_ETH
#SBATCH --output=DPSGD_ETH.out
#SBATCH --error=DPSGD_ETH.err
#SBATCH --nodes=NB_NODES
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:8
#SBATCH --time=30:00:00

# Replace NB_NODES with the number of nodes to use
# Load any modules and activate your conda environment here

# the 'resume' argument is passed into this bash script
resume_from_checkpoint=$1
if [ -z "$resume_from_checkpoint" ]; then
    echo "Warning: resume arg not set (defaulting to False)"
    resume_from_checkpoint=False
fi

srun python -u gossip_sgd.py --resume $resume_from_checkpoint \
    --batch_size 256 --distributed True --lr 0.1 --num_dataloader_workers 16 \
    --num_epochs 90 --nesterov True --warmup True --push_sum False \
    --graph_type 1 --schedule 30 0.1 60 0.1 80 0.1 --backend 'tcp'  \
    --train_fast False --master_port 40100  \
    --tag 'DPSGD_ETH' --print_freq 100 --verbose False \
    --single_threaded False --overlap False \
    --all_reduce False --seed 1
