# Train a Transformer with Stochastic Gradient Push

This directory contains the code to reproduce the neural machine translation experiment reported in the paper
> Anonymous Authors, "Stochastic Gradient Push for Distributed Deep Learning," submitted to ICML 2019.

## Setup
In addition to PyTorch and other dependencies listed in [../README.md](../README.md):
1. Checkout the [fairseq github repository](https://github.com/pytorch/fairseq) at commit 8eb232ce150d1afb44880a7078eb4abbae60dc32
2. Copy the entire directory `../gossip_module/` into the root repo directory
3. Replace the following files in the root repo directory with the ones here:
```distributed_train.py
fairseq/distriuted_utils.py
fairseq/models/distributed_fairseq_model.py
fairseq/options.py
fairseq/trainer.py
fairseq/utils.py
train.py
```
4. Download and preprocess the WMT'16 En-De data following the instructions (steps 1 and 2 only) [from the fairseq documentation](https://github.com/pytorch/fairseq/tree/master/examples/translation#replicating-results-from-scaling-neural-machine-translation)
5. Copy the directory `./job_scripts/` into the root repo directory

## Launching experiments

To facilitate reproducibility, we provide example scripts for submitting jobs using the SLURM workload manager. Note that these will only be directly usable if your cluster also uses SLURM, but hopefully they will be useful, regardless, as examples of how to reproduce our results.

The `./job_scripts/` directory contains four files:
1. `submit_ar_small.sh` runs AllReduce with small (25k token) batch size
2. `submit_sgp_small.sh` runs SGP with small (25k token) batch size
3. `submit_ar_large.sh` runs AllReduce with large (400k token) batch size
4. `submit_sgp_large.sh` runs SGP with large (400k token) batch size
