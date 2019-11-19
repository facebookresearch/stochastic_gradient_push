# Stochastic Gradient Push
#### *Gossip-based distributed optimization algorithms implemented in PyTorch*

Code to reproduce the experiments reported in the paper:
> Mido Assran, Nicolas Loizou, Nicolas Ballas, and Michael Rabbat, "Stochastic Gradient Push for Distributed Deep Learning," ICML 2019. [Official ICML version](http://proceedings.mlr.press/v97/assran19a.html) [arxiv version](https://arxiv.org/abs/1811.10792)

If you use this code for your research, please cite the paper.

It implements the following algorithms:
* Synchronous Stochastic Gradient Push (SGP), described in the paper
* Overlap Stochastic Gradient Push (OSGP), described in the paper
* AllReduce SGD (AR), standard baseline, also known as Parallel SGD, implemented using PyTorch's `torch.nn.parallel.DistributedDataParallel`
* Distributed Parallel SGD (D-PSGD), described in [Lian et al., NeurIPS 2017](https://arxiv.org/abs/1705.09056)
* Asynchronous Distributed Parallel SGD (AD-PSGD), described in [Lian et al., ICML 2018](https://arxiv.org/abs/1710.06952)

An example is provided for training a ResNet-50 [(He et al., 2015)](https://arxiv.org/abs/1512.03385) image classifier on the [ImageNet dataset](http://www.image-net.org/).

## Dependencies and Setup
All code runs on Python 3.6.7 using [PyTorch version 1.0.0](https://github.com/pytorch/pytorch/tree/v1.0.0).

Our implementations build on the `torch.distributed` package in PyTorch, which provides an interface for exchanging tensors between multiple machines. The `torch.distributed` package in PyTorch v.1.0.0 can use different backends. We recommmend using NCCL for all algorithms (this is the default).

* You could run `./scripts/setup.sh` to install all the dependencies. You may have to change the CUDA version etc  (in the `requirements.txt` file) to match your setup.

In addition, you will need to install
* [torchvision 0.2.1](https://github.com/pytorch/vision/tree/v0.2.1) for the ResNet-50 model
* matplotlib 2.2.2 for producing figures
* pandas 0.24.1 for loading results from CSV files
* [apex](https://github.com/NVIDIA/apex/tree/1b903852aecd388e10f03e470fcb1993f1c871dd) for fp16

## Running Experiments
### Training ResNet-50 on ImageNet
There are two main scripts:
* `gossip_sgd.py` for training using AR, SGP, OSGP, or D-PSGD
* `gossip_sgd_adpsgd.py` for training using AD-PSGD

In order to facilitate launching experiments, we also provide example scripts for submitting jobs using the SLURM workload manager. Note that these will only be directly usable if your cluster also uses SLURM, but hopefully they will be useful, regardless, as examples of how to launch distributed jobs.

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

## Reproducing figures in the paper
Note that the current version in the master branch of this repo uses features introduced in PyTorch 1.0. The version of the code used to produce the results in the paper was based on PyTorch 0.5. That version of our code is available under the `sgp_pytorch0.5` tag of this repo.

Figures similar to those in the paper can be reproduced, after running the experiments to generate log files, using the script `visualization/plotting.py`. This script will also need to be modified to use the same paths to log files you used when running the experiments.

## Overview of the implementation, code organization
### Training neural networks
The algorithms SGP, D-PSGD, and AD-PSGD are all implemented as instances of PyTorch's `nn.Module` class to facilitate training neural network models. SGP and D-PSGD are implemented in the `GossipDataParallel` class in `gossip_modules/distributed.py`. The `push_sum` argument determines whether to use SGP (if `push_sum=True`) or D-PSGD (if `push_sum=False`). Overlap SGP is obtained by using the `GossipDataParallel` class with `push_sum=True` and `overlap=True`. AD-PSGD is implemented in the `BilatGossipDataParallel` class in `gossip_modules/ad_psgd.py`

### Gossip-based distributed averaging
The neural network modules use implementations of PushSum and gossip algorithms for distributed averaging under the hood. These are availble in `gossip_modules/gossiper.py` and could be used independently of neural network training for approximate distributed averaging. In addition:
* `gossip_modules/graph_manager.py` contains code to generate different communication topologies, and
* `gossip_modules/mixing_manager.py` contains code to produce weights of the mixing matrices, given a topology.

## License
See the [LICENSE](./LICENSE) file for details about the license under which this code is made available.
