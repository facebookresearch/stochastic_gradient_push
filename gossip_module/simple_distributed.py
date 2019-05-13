# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
Simple Distributed Gossip Wrapper

:description: Single-Threaded Gossip Model Wrapper; designed for efficient
              multi-peer communication. When sending and receiving more than
              one message in each iteration, it is preferable to use a
              multi-threaded gossip model wrapper.
"""

import time

import torch
import torch.distributed as dist
from torch.cuda.comm import broadcast_coalesced, reduce_add_coalesced
from torch.autograd import Variable
from torch.nn.modules import Module
from torch.nn.parallel.replicate import replicate
from torch.nn.parallel.scatter_gather import scatter_kwargs, gather
from torch.nn.parallel.parallel_apply import parallel_apply

from .gossiper import PushSum, PushPull
from .utils import make_logger, _flatten_tensors, _unflatten_tensors
from .utils.metering import Meter


class SimpleGossipDataParallel(Module):

    def __init__(self, module, device_ids=None, distributed=True,
                 graph=None, comm_device=None, push_sum=True,
                 verbose=True):
        super(SimpleGossipDataParallel, self).__init__()

        # whether we're using multiple agents for training
        self.distributed = distributed

        # devices available locally
        if device_ids is None:
            device_ids = list(range(torch.cuda.device_count()))
        self.device_ids = device_ids
        self.output_device = self.device_ids[0]

        # put model on output device
        self.module = module.cuda(self.output_device)

        # prepare local intra-node all-reduce objects
        if len(self.device_ids) > 1:
            self.broadcast_bucket_size = 10 * 1024 * 1024  # bytes
            self.nccl_reduce_bucket_size = 256 * 1024 * 1024  # bytes

            self._module_copies = replicate(self.module, self.device_ids,
                                            detach=True)
            self._module_copies[0] = self.module
            for cmodule in self._module_copies[1:]:
                for p, cp in zip(self.module.parameters(),
                                 cmodule.parameters()):
                    cp.requires_grad = p.requires_grad
        else:
            self._module_copies = [self.module]

        # prepare inter-node gossip objects
        if self.distributed:
            assert dist.is_initialized()

            # set distributed configuration properties
            self.graph = graph
            self.push_sum = push_sum
            self.gossip_enable = True
            if comm_device is None:
                comm_device = torch.device('cpu')
            self.comm_device = comm_device

            # logger used to print to stdout
            self.logger = make_logger(dist.get_rank(), verbose)

            # initalize gossiper to push-sum or push-pull protocol
            if self.push_sum:
                self.gossiper = PushSum(
                    msg=_flatten_tensors(list(self.module.parameters())),
                    device=self.comm_device,
                    graph=self.graph,
                    logger=self.logger)
            else:
                self.gossiper = PushPull(
                    msg=_flatten_tensors(list(self.module.parameters())),
                    device=self.comm_device,
                    graph=self.graph,
                    logger=self.logger)
        else:
            # logger used to print to stdout
            self.logger = make_logger(0, verbose)

        # register hook for gradient reduction on all GPUs avaialable locally
        self.register_backward_hook(self.__make_backward_hook())

    def forward(self, *inputs, **kwargs):
        """ Forward pass performed in parallel across all devices on node """
        # put inputs onto devices
        inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids)
        if len(self.device_ids) > 1:
            # run forward pass across all devices
            self._sync_params()
            outputs = self.parallel_apply(self._module_copies[:len(inputs)],
                                          inputs, kwargs)
            return self.gather(outputs, self.output_device)
        else:
            return self.module(*inputs[0], **kwargs[0])

    def scatter(self, inputs, kwargs, device_ids):
        return scatter_kwargs(inputs, kwargs, device_ids, dim=0)

    def parallel_apply(self, replicas, inputs, kwargs):
        return parallel_apply(replicas, inputs, kwargs,
                              self.device_ids[:len(replicas)])

    def gather(self, outputs, output_device):
        return gather(outputs, output_device, dim=0)

    def _sync_params(self):
        """ Synchronoize parameters across devices (intra-node) """
        if len(self.device_ids) <= 1:
            return

        # intra-node parameter sync
        params = [p.data for p in self.module.parameters()]
        result = broadcast_coalesced(params, self.device_ids,
                                     self.broadcast_bucket_size)
        for tensors, module in zip(result[1:], self._module_copies[1:]):
            for tensor, param in zip(tensors, module.parameters()):
                param.data.set_(tensor)

        # intra-node buffer sync
        buffers = [b.data for b in self.module._all_buffers()]
        if len(buffers) > 0:
            result = broadcast_coalesced(buffers, self.device_ids,
                                         self.broadcast_bucket_size)
            for tensors, module in zip(result[1:], self._module_copies[1:]):
                for tensor, buf in zip(tensors, module._all_buffers()):
                    buf.data.set_(tensor)

    def train(self, mode=True):
        super(SimpleGossipDataParallel, self).train(mode)
        self.gossip_enable = True
        for module in self._module_copies[1:]:
            module.train(mode)

    def eval(self):
        super(SimpleGossipDataParallel, self).eval()
        self.gossip_enable = False
        for module in self._module_copies[1:]:
            module.eval()

    def block(self):
        # cannot block if program not distributed
        if not self.distributed:
            return
        self.logger.info('blocking')
        dist.barrier()

    def transfer_params(self):
        # don't gossip if program not distributed
        if not self.distributed:
            return
        self.logger.debug('gossiping params')
        out_msg = _flatten_tensors(list(self.module.parameters())).detach_()
        in_msg, _ = self.gossiper.mix(out_msg, 1.0, residual=True)
        for g, p in zip(_unflatten_tensors(in_msg,
                                           list(self.module.parameters())),
                        self.module.parameters()):
            p.data.add_(g).mul_(0.5)

    def __make_backward_hook(self):
        self.logger.debug('making backward hook')

        def hook(*unused):
            # reduce gradients across devices on a single machine
            if len(self.device_ids) > 1:

                # collect gradients from all copies
                all_grads = [[] for _ in range(len(self._module_copies))]
                for dev_idx, module in enumerate(self._module_copies):
                    for p in module.parameters():
                        if not p.requires_grad or p.grad is None:
                            continue
                        all_grads[dev_idx].append(p.grad.data)

                # reduce grads
                reduced_grads = reduce_add_coalesced(
                    all_grads, self.output_device,
                    self.nccl_reduce_bucket_size)

                # update grads with reduced grads
                for grad, reduced in zip(all_grads[0], reduced_grads):
                    grad.copy_(reduced)

                # clear the gradients and parameters across all replicas
                for module in self._module_copies[1:]:
                    for param in module.parameters():
                        if param.requires_grad:
                            param.grad = None
                            param.data.set_()

        def queue_hook(*unused):
            Variable._execution_engine.queue_callback(hook)
        return queue_hook

    def communicator_warmup(self):
        """ time the push-sum code """
        if not self.distributed:
            return
        dist.barrier()
        # test push-sum
        gossip_tensor = _flatten_tensors(list(self.module.parameters()))
        gossip_tensor.zero_()
        gossip_tensor += dist.get_rank()
        gossip_tensor = gossip_tensor.detach_().to(self.comm_device)
        gossiper = PushSum(gossip_tensor, logger=self.logger)
        gossip_meter = Meter(ptag='Gossip', stateful=True, csv_format=False)
        self.logger.debug('testing push-sum')
        self.logger.debug('push-sum begin {}/{}'.format(
            gossip_tensor[0].item(),
            gossip_tensor[-1].item()))
        for _ in range(100):
            bt = time.time()
            # mix tensor
            ps_factor = 0.5
            gossip_tensor.mul_(ps_factor)
            residual = gossip_tensor.clone()
            residual, _ = gossiper.mix(residual, 1.0, residual=True)
            gossip_tensor.add_(residual)
            gossip_meter.update(time.time() - bt)
            self.logger.debug(gossip_meter)
        dist.barrier()
        self.logger.debug('push-sum result {}/{}'.format(
            gossip_tensor[0].item(),
            gossip_tensor[-1].item()))
        gossiper.clean_msg_buffers_()
        dist.barrier()
        del gossiper
        del gossip_meter
        del gossip_tensor
