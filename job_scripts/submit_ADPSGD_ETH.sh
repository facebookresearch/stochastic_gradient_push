#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
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
    --batch_size 256 --lr 0.1 --num_dataloader_workers 16 \
    --num_epochs 90 --nesterov True --warmup True --push_sum False \
    --graph_type 1 --schedule 30 0.1 60 0.1 80 0.1 \
    --train_fast True --master_port 40100 --network_interface_type 'ethernet' \
    --tag 'ADPSGD_ETH' --print_freq 100 --verbose False --bilat True \
    --all_reduce False --seed 1 \
    --shared_fpath 'results_dir/itr32.txt'
