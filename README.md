# Stochastic Gradient Push
#### *Gossip-based distributed optimization algorithms implemented in PyTorch*

Code to reproduce the experiments reported in the paper:
> Mido Assran, Nicolas Loizou, Nicolas Ballas, and Michael Rabbat, "Stochastic Gradient Push for Distributed Deep Learning," ICML 2019.

If you use this code for your research, please cite the paper.

It implements the following algorithms:
* Synchronus Stochastic Gradient Push (SGP), described in the paper
* Overlap Stochastic Gradient Push (OSGP), described in the paper
* AllReduce SGD (AR), standard baseline, also known as Parallel SGD
* Distributed Parallel SGD (D-PSGD), described in [Lian et al., NeurIPS 2017](https://arxiv.org/abs/1705.09056)
* Asynchronous Distributed Parallel SGD (AD-PSGD), described in [Lian et al., ICML 2018](https://arxiv.org/abs/1710.06952)

For two tasks:
1. Training a ResNet-50 [(He et al., 2015)](https://arxiv.org/abs/1512.03385) image classifier on the [ImageNet dataset](http://www.image-net.org/)
2. Training a transformer neural machine translation model [(Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762) on the [WMT'16 En-De dataset](http://www.statmt.org/wmt16/translation-task.html)

## Dependencies and Setup
All code runs on Python 3.6.5 using [PyTorch version 0.5.0a0+ec42a11](https://github.com/pytorch/pytorch/tree/ec42a1141083f1266c079756c96df287c965b18e).

Our implementations build on the `torch.distributed` package in PyTorch, which provides an interface for exchanging tensors between multiple machines. The `torch.distributed` package in PyTorch v.0.5.0 can use different backends, and the ones we used in our experiments are:
* TCP for all methods (gossip-based and AllReduce) over Ethernet
* MPI for gossip-based methods over InfiniBand
* Gloo for AllReduce on InfiniBand (*Note:* We observed similar performance using the Gloo and NCCL backends over InfiniBand)

To use the MPI backend you will need to have a version of MPI installed on your cluster, and you need to compile PyTorch with MPI support following [these instructions](https://github.com/pytorch/pytorch/tree/ec42a1141083f1266c079756c96df287c965b18e#from-source). The experiments reported in the paper were run using OpenMPI 3.0.0.

In addition, you will need to install
* [torchvision 2.0.1](https://github.com/pytorch/vision/tree/v0.2.1) for the ResNet-50 model
* matplotlib 2.2.2 for producing figures
* pandas 0.23.1 for loading results from CSV files

## Running Experiments
### Training ResNet-50 on ImageNet
There are two main scripts:
* `gossip_sgd.py` for training using AR, SGP, OSGP, or D-PSGD
* `gossip_sgd_arsgd.py` for training using AD-PSGD

In order to facilitate reproducing our experiments, we also provide example scripts for submitting jobs using the SLURM workload manager. Note that these will only be directly usable if your cluster also uses SLURM, but hopefully they will be useful, regardless, as examples of how to reproduce our results.

The `job_scripts/` directory contains the following files:
* `submit_ADPSGD_ETH.sh` runs the AD-PSGD algorithm over Ethernet
* `submit_AR_ETH.sh` runs the AR algorithm over Ethernet
* `submit_AR_IB.sh` runs the AR algorithm over InfiniBand
* `submit_DPSGD_ETH.sh` runs the D-PSGD algorithm over Ethernet
* `submit_DPSGD_IB.sh` runs the D-PSGD algorithm over InfiniBand
* `submit_SGP_ETH.sh` runs the SGP algortihm over Ethernet
* `submit_SGP_IB.sh` runs the SGP algorithm over InfiniBand

In all cases, the scripts will need to be editied/modified in order to run on your cluster/setup. They also contain instructions on how to modify the script, e.g., to vary the number of nodes or other parameters.

The SGP scripts currently implement Synchronous SGP. To run experiments for Overlap SGP (overlapping communication and computation), change the `--overlap` flag to `True`.

### Training a Transformer
See the code and [Readme.md](./transformer/Readme.md) in the `transformer/` directory.

## Reproducing figures in the paper
All figures in the paper can be reproduced, after running the experiments to generate log files, using the script `visualization/plotting.py`. This script will also need to be modified to use the same paths to log files you used when running the experiments.

## Overview of the implementation, code organization
### Training neural networks
The algorithms SGP, AllReduce SGD, D-PSGD, and AD-PSGD are all implemented as instances of PyTorch's `nn.Module` class to facilitate training neural network models.
* AD-PSGD is implemented in the `BilatGossipDataParallel` class in `gossip_modules/ad_psgd.py`
* AllReduce SGD is implemented in the `AllReduceDataParallel` class in `gossip_modules/ar_distributed.py`

In addition, we provide single- and multi-threaded implementations of D-PSGD and SGP. The single-threaded implementations are available as the `SimpleGossipDataParallel` class in `gossip_modules/simple_distributed.py`, and the multi-threaded implementations are avaialable as the `GossipDataParallel` class in `gossip_modules/distributed.py`. In both cases, the `push_sum` argument determines whether to use SGP (if `push_sum=True`) or D-PSGD (if `push_sum=False`). Overlap SGP is obtained by using the `GossipDataParallel` class with `push_sum=True` and `overlap=True`.

### Gossip-based distributed averaging
The neural network modules use implementations of PushSum and gossip algorithms for distributed averaging under the hood. These are availble in `gossip_modules/gossiper.py` and could be used independently of neural network training for approximate distributed averaging. In addition:
* `gossip_modules/graph_manager.py` contains code to generate different communication topologies, and
* `gossip_modules/mixing_manager.py` contains code to produce weights of the mixing matrices, given a topology.

