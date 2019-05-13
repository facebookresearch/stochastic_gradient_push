#!/bin/bash
#SBATCH --job-name=SGP-ETH
#SBATCH --output=SGP-ETH.out
#SBATCH --error=SGP-ETH.err
#SBATCH --nodes=NB_NODES
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:8
#SBATCH --time=05:00:00

# Replace NB_NODES with the number of nodes to use

# Load any modules and activate your conda environment here

srun python -u gossip_sgd.py \
    --batch_size 256 --distributed True --lr 0.1 --num_dataloader_workers 16 \
    --num_epochs 90 --nesterov True --warmup True --push_sum True \
    --graph_type 0 --schedule 30 0.1 60 0.1 80 0.1 --backend 'tcp'  \
    --train_fast False --master_port 40100 \
    --tag 'SGP_ETH' --print_freq 100 --verbose False \
    --single_threaded False --overlap False
