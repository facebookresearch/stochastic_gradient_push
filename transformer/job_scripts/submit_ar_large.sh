#!/bin/bash
#SBATCH --job-name=transformer_test
#SBATCH --output=transformer_ar_test.out
#SBATCH --error=transformer_ar_test.err
#SBATCH --nodes=8
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00
#SBATCH --open-mode=append

# Load any modules and activate your conda environment here
srun --unbuffered --label python -u train.py data-bin/wmt16_en_de_bpe32k \
	--max-tokens 3500 --distributed-port 40100 --ddp-backend no_c10d \
	--dist_avg 'allreduce2' --distributed-backend tcp --update-freq 16 \
	--arch transformer_vaswani_wmt_en_de_big --share-all-embeddings \
	--optimizer adam --adam-betas '(0.9, 0.98)' \
	--lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 4000 \
	--lr 0.0005 --min-lr 1e-09 --clip-norm 0.0 \
	--dropout 0.3 --weight-decay 0.0 --criterion label_smoothed_cross_entropy \
	--label-smoothing 0.1 --log-format simple \
	--save-dir out_files/ar_test
