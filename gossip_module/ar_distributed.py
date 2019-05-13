# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
All-Reduce Distributed Model Wrapper

"""
import time
import sys
import threading

import torch
import torch.distributed as dist
from torch.cuda.comm import broadcast_coalesced, reduce_add_coalesced
from torch.autograd import Variable
from torch.nn.modules import Module
from torch.nn.parallel.replicate import replicate
from torch.nn.parallel.scatter_gather import scatter_kwargs, gather
from torch.nn.parallel.parallel_apply import parallel_apply

from .utils import make_logger, _flatten_tensors, _unflatten_tensors
from .utils.metering import Meter

HEARTBEAT_TIMEOUT = 300  # seconds


class AllReduceDataParallel(Module):
    """ Distributed Gossip model wrapper """

    def __init__(self, module, device_ids=None, distributed=True,
                 world_size=None, rank=None, comm_device=None,
                 verbose=True):
        super(AllReduceDataParallel, self).__init__()

        # whether we're using multiple agents for training
        self.distributed = distributed

        # devices available locally
        if device_ids is None:
            device_ids = list(range(torch.cuda.device_count()))
        self.output_device = device_ids[0]
        self.device_ids = device_ids

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
            if world_size is None or rank is None:
                assert dist.is_initialized()
                world_size = dist.get_world_size()
                rank = dist.get_rank()

            # communicate over cpu's if not specified
            if comm_device is None:
                comm_device = torch.device('cpu')
            self.__cpu_comm = comm_device.type == 'cpu'

            # distributed backend config
            self.dist_config = {
                'verbose': verbose,
                'comm_device': comm_device,
                'rank': rank
            }

            # logger used to print to stdout
            self.logger = make_logger(rank, verbose)

            # prepare parameters for gossip
            self.ps_factor = 1. / world_size
            self.gossip_enable = True
            self.gossiping = False
            self.params_mixed = True
            self.gossip_params = []
            self.gossip_device_buffer = []
            for p in module.parameters():
                cp = p.clone().detach_()
                cp = cp.cpu().pin_memory() if self.__cpu_comm else cp.cuda()
                self.gossip_params.append(cp)

            # prepare gossip process control objects
            self.gossip_flag = threading.Event()
            self.train_flag = threading.Event()
            self.gossip_thread = threading.Thread(
                target=AllReduceDataParallel._gossip_target,
                args=(self.dist_config,
                      self.gossip_flag,
                      self.train_flag,
                      self.gossip_params,
                      self.gossip_device_buffer))
            self.gossip_thread.daemon = True
            self.gossip_thread.name = 'Gossip-Thread'
            self.gossip_thread.start()
            # wait for thread to complete initialization
            self.gossip_flag.wait()
            self.gossip_flag.clear()
        else:
            self.params_mixed = True
            # logger used to print to stdout
            self.logger = make_logger(0, verbose)

        # register grad-reduction hooks
        self.__register_hooks()

    def forward(self, *inputs, **kwargs):
        """ Forward pass performed in parallel across all devices on node """
        # scatter inputs onto devices
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
        super(AllReduceDataParallel, self).train(mode)
        self.gossip_enable = True
        for module in self._module_copies[1:]:
            module.train(mode)

    def eval(self):
        super(AllReduceDataParallel, self).eval()
        self.gossip_enable = False
        for module in self._module_copies[1:]:
            module.eval()
        if self.distributed:
            self._query_gossip_queue()

    def block(self):
        if not self.distributed:
            return
        self.logger.info('blocking')
        dist.barrier()

    def _query_gossip_queue(self, non_blocking=False):
        """ Check gossip-queue for push-sum residuals and update model """
        if not self.distributed:
            return

        self.logger.debug('querying gossip queue')

        # no gossip happening right now so just return
        if not self.gossiping:
            self.logger.warning('not gossiping right now')
            return False

        if not non_blocking:
            if not self.gossip_flag.wait(timeout=HEARTBEAT_TIMEOUT):
                sys.exit()  # HEARTBEAT monitor

        # query gossip thread
        if self.gossip_flag.is_set():
            self.logger.debug('received gossip flag')

            for p, r in zip(self.module.parameters(),
                            self.gossip_device_buffer):
                p.data.set_(r)

            # update flags
            self.logger.debug('updated model params')
            self.gossip_device_buffer.clear()
            self.gossip_flag.clear()
            self.params_mixed = True
            self.gossiping = False
            return True

    def transfer_params(self, mix=True):
        """ Transfers COPY of model parameters to gossip queue """
        if not self.distributed:
            return False

        self.logger.debug('transfering model params')

        # don't transfer new params if old params haven't been mixed yet
        if not self.params_mixed:
            self.logger.warning('params not mixed')
            return False

        # Transfer params to gossip-process:
        # --
        # params gpu-gpu copy (fast)
        # --
        for p in self.module.parameters():
            if mix:
                p.data.mul_(self.ps_factor)
            self.gossip_device_buffer.append(p.clone().detach_())
        # --
        # buffer to gossip-thread copy (potentially slow, but asynchronous)
        # --
        for b, gp in zip(self.gossip_device_buffer, self.gossip_params):
            if self.__cpu_comm:
                gp.copy_(b, non_blocking=True)
            else:
                gp.data.set_(b)
        # --

        # update flags
        self.logger.debug('transfered model params')
        self.params_mixed = False
        self.gossiping = True
        self.train_flag.set()
        return True

    @staticmethod
    def _gossip_target(dist_config, gossip_flag, train_flag,
                       gossip_params, gossip_device_buffer):
        """ Gossip thread, which performs push-sum on model params """
        logger = make_logger(dist_config['rank'], dist_config['verbose'])

        if dist_config['comm_device'].type != 'cpu':
            gossip_stream = torch.cuda.Stream()
            dist._register_stream(gossip_stream)
        else:
            gossip_stream = torch.cuda.current_stream()

        gossip_flag.set()

        # gossip loop
        while True:
            train_flag.wait()
            logger.debug('received train-flag')
            try:
                with torch.cuda.stream(gossip_stream):
                    # construct gossip tensor
                    out_msg = _flatten_tensors(gossip_params)
                    dist.all_reduce(out_msg)
                    # update gossip variables with result
                    for r, g in zip(_unflatten_tensors(out_msg, gossip_params),
                                    gossip_device_buffer):
                        g.copy_(r, non_blocking=True)
            except RuntimeError as e:
                logger.warning('received runtime error {}'.format(e))
            finally:
                # give main thread go-ahead to read our gossip buffer
                train_flag.clear()
                gossip_flag.set()

    def __register_hooks(self):
        """
        Registers gossip/all-reduce hooks in pre-forward/post-backward pass
        """
        self.register_forward_pre_hook(self.__make_forward_pre_hook())
        self.register_backward_hook(self.__make_backward_hook())

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

    def __make_forward_pre_hook(self):
        self.logger.debug('making forward pre-hook')

        def hook(*unused):
            """ Query gossip queue during forward pass """
            if self.distributed:
                # gossip during training (not inference)
                if self.gossip_enable:
                    self._query_gossip_queue()

        return hook

    def communicator_warmup(self):
        """ time the all-reducde code """
        if not self.distributed:
            return
        dist.barrier()
        time.sleep(5)
        dist.barrier()
