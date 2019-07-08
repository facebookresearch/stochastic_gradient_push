#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#SBATCH --job-name=DPSGD_IB
#SBATCH --output=DPSGD_IB.out
#SBATCH --error=DPSGD_IB.err
#SBATCH --nodes=NB_NODES
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:8
#SBATCH --time=30:00:00

# Replace NB_NODES with the number of nodes to use
# Load any modules and activate your conda environment here

srun python -u gossip_sgd.py \
    --batch_size 256 --lr 0.1 --num_dataloader_workers 16 \
    --num_epochs 90 --nesterov True --warmup True --push_sum False \
    --graph_type 1 --schedule 30 0.1 60 0.1 80 0.1 \
    --train_fast False --master_port 40100 \
    --tag 'DPSGD_IB' --print_freq 100 --verbose False \
    --all_reduce False --seed 1 \
    --network_interface_type 'infiniband'
