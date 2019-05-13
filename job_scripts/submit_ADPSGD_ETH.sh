#!/bin/bash
#SBATCH --job-name=ADPSGD_ETH
#SBATCH --output=ADPSGD_ETH.out
#SBATCH --error=ADPSGD_ETH.err
#SBATCH --nodes=NB_NODES
#SBATCH --cpus-per-task=80
#SBATCH --gres=gpu:8
#SBATCH --time=30:00:00

# Replace NB_NODES above with the number of nodes to use

# Load any modules and activate your conda environment here

srun python -u gossip_sgd_adpsgd.py \
    --batch_size 256 --distributed True --lr 0.1 --num_dataloader_workers 16 \
    --num_epochs 90 --nesterov True --warmup True --push_sum False \
    --graph_type 1 --schedule 30 0.1 60 0.1 80 0.1 --backend 'tcp'  \
    --train_fast True --master_port 40100 \
    --tag 'ADPSGD_ETH' --print_freq 50 --verbose True --bilat True \
    --single_threaded False --overlap False --all_reduce False \
    --shared_fpath 'results_dir/itr32.txt'
