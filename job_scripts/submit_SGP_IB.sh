#!/bin/bash
#SBATCH --job-name=SGP_IB
#SBATCH --output=SGP_IB.out
#SBATCH --error=SGP_IB.err
#SBATCH --nodes=NB_NODES
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:8
#SBATCH --time=30:00:00

# Replace NB_NODES with the number of nodes to use

# Load any modules and activate your conda environment here

mpirun --mca btl_openib_cuda_async_recv false \
    --mca coll_tuned_allreduce_algorithm 5 \
    --mca opal_abort_print_stack true --mca btl_openib_want_cuda_gdr true \
    --mca mpi_preconnect_all true --mca opal_leave_pinned true \
    python -u gossip_sgd.py \
    --batch_size 256 --distributed True --lr 0.1 --num_dataloader_workers 16 \
    --num_epochs 90 --nesterov True --warmup True --push_sum True \
    --graph_type 0 --schedule 30 0.1 60 0.1 80 0.1 --backend 'mpi'  \
    --train_fast False --master_port 40100 \
    --tag 'SGP_IB' --print_freq 100 --verbose False \
    --single_threaded True --overlap False
