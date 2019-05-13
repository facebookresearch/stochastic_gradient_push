# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
Distributed Gossip Wrapper

:description: Multi-Threaded Gossip Model Wrapper; designed for efficient
              multi-peer training.
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

from .gossiper import PushSum, PushPull
from .utils import make_logger, _flatten_tensors, _unflatten_tensors

HEARTBEAT_TIMEOUT = 300  # maximum time to wait for message (seconds)


class GossipDataParallel(Module):
    """ Distributed Gossip model wrapper """

    def __init__(self, module, device_ids=None, distributed=True,
                 graph=None, mixing=None, comm_device=None, push_sum=True,
                 rank=None, world_size=None,
                 overlap=False, synch_freq=0, verbose=True):
        super(GossipDataParallel, self).__init__()

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
                rank = dist.get_rank()
                world_size = dist.get_world_size()

            # communicate over cpu's if not specified
            if comm_device is None:
                comm_device = torch.device('cpu')
            self.__cpu_comm = comm_device.type == 'cpu'

            # distributed backend config
            self.dist_config = {
                'verbose': verbose,
                'comm_device': comm_device,
                'graph': graph,
                'mixing': mixing,
                'push_sum': push_sum,
                'rank': rank,
                'world_size': world_size
            }
            self.overlap = overlap
            self.synch_freq = synch_freq
            self.num_updates = 0
            self.asynch = synch_freq > 0

            # logger used to print to stdout
            self.logger = make_logger(rank, verbose)

            # push-sum weight=1.0 ==> distributed averaging
            self.ps_weight = 1.0
            self.is_ps_numerator = False

            # prepare parameters for gossip
            self.gossip_enable = True
            self.gossiping = False
            self.params_mixed = True
            self.gossip_ps_factor = [None]
            self.gossip_ps_weight = [self.ps_weight]
            self.gossip_params = []
            self.gossip_device_buffer = []
            for p in module.parameters():
                cp = p.clone().detach_()
                cp = cp.cpu().pin_memory() if self.__cpu_comm else cp.cuda()
                self.gossip_params.append(cp)

            # prepare gossip process control objects
            self.gossip_lock = threading.Lock()
            self.gossip_flag = threading.Event()
            self.train_flag = threading.Event()
            self.gossip_thread = threading.Thread(
                target=GossipDataParallel._gossip_target,
                args=(self.dist_config,
                      self.gossip_flag,
                      self.train_flag,
                      self.gossip_lock,
                      self.gossip_params,
                      self.gossip_device_buffer,
                      self.gossip_ps_weight,
                      self.gossip_ps_factor))
            self.gossip_thread.daemon = True
            self.gossip_thread.name = 'Gossip-Thread'
            self.gossip_thread.start()
            # wait for thread to complete initialization
            self.gossip_flag.wait()
            self.gossip_flag.clear()
            # lazy mixing avoids additional bias/de-bias steps
            self.lazy_mixing = not self.asynch \
                and self.dist_config['mixing'].is_regular()
            self.lazy_ps_factor = self.gossip_ps_factor[0]
            self.logger.debug('lazy mixing: {}'.format(self.lazy_mixing))
        else:
            self.params_mixed = True
            # logger used to print to stdout
            self.logger = make_logger(0, verbose)

        # register ps/grad-reduction hooks
        self.__register_hooks()

    def update_gossiper(self, attr, val):
        if not self.distributed:
            return
        self.logger.debug('waiting for gossip lock')
        with self.gossip_lock:
            self.logger.debug('gossip lock received')
            if val == getattr(self.dist_config['gossiper'], attr):
                self.logger.debug('nothing to update')
                return
            # update attr
            self.logger.debug('setting gossiper {} to {}'.format(attr, val))
            setattr(self.dist_config['gossiper'], attr, val)

    def state_dict(self):
        super_dict = super(GossipDataParallel, self).state_dict()
        if not self.distributed:
            return {'state_dict': super_dict}
        supplanted_dict = {'state_dict': super_dict,
                           'ps_weight': self.ps_weight,
                           'is_ps_numerator': self.is_ps_numerator}
        return supplanted_dict

    def load_state_dict(self, load_dict):
        state_dict = load_dict['state_dict']
        super(GossipDataParallel, self).load_state_dict(state_dict)
        if not self.distributed:
            return
        self.ps_weight = load_dict['ps_weight']
        self.is_ps_numerator = load_dict['is_ps_numerator']

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

    def ps_numerator(self):
        """ Convert model params to ps-numerator """
        if not self.distributed:
            return
        if not self.is_ps_numerator:
            ps_weight = self.ps_weight
            if ps_weight != 1. and not self.lazy_mixing:
                for p in self.module.parameters():
                    p.data.mul_(ps_weight)
            self.is_ps_numerator = True

    def unbias(self):
        """ Convert moel params to de-biased estimate """
        if not self.distributed:
            return
        if self.is_ps_numerator:
            ps_weight = self.ps_weight
            if ps_weight != 1. and not self.lazy_mixing:
                for p in self.module.parameters():
                    p.data.div_(ps_weight)
            self.is_ps_numerator = False

    def train(self, mode=True):
        super(GossipDataParallel, self).train(mode)
        self.gossip_enable = True
        for module in self._module_copies[1:]:
            module.train(mode)

    def eval(self):
        super(GossipDataParallel, self).eval()
        self.gossip_enable = False
        for module in self._module_copies[1:]:
            module.eval()
        if self.distributed:
            self._query_gossip_queue(non_blocking=self.asynch)

    def block(self):
        if not self.distributed:
            return
        self.logger.info('blocking')
        dist.barrier()

    def sync_comms(self):
        if not self.distributed:
            return
        self._query_gossip_queue(non_blocking=False)

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

            # atomic gossip was interrupted so try again
            if self.gossip_ps_weight[0] == -1:
                self.gossip_device_buffer.clear()
                self.gossip_flag.clear()
                self.params_mixed = True
                self.gossiping = False
                self.transfer_params(mix=False)
                return False

            self.lazy_ps_factor = self.gossip_ps_factor[0]

            # convert model-params to ps numerators b4 adding residuals
            self.ps_numerator()

            # add residuals
            self.ps_weight += self.gossip_ps_weight[0]
            if self.lazy_mixing:
                self.ps_weight *= self.lazy_ps_factor
            for p, r in zip(self.module.parameters(),
                            self.gossip_device_buffer):
                p.data.add_(r)
                if self.lazy_mixing:
                    p.data.mul_(self.lazy_ps_factor)

            # update flags
            self.logger.debug('updated ps-weight {}'.format(self.ps_weight))
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

        # using lazy mixing ==> mix on query not transfer
        mix = mix and not self.lazy_mixing

        # Transfer ps-numerators to gossip-process:
        # --
        self.ps_numerator()
        if mix:
            self.ps_weight *= self.gossip_ps_factor[0]
        self.gossip_ps_weight[0] = self.ps_weight
        # --
        # params gpu-gpu copy (fast)
        # --
        for p in self.module.parameters():
            if mix:
                p.data.mul_(self.gossip_ps_factor[0])
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
    def _gossip_target(dist_config, gossip_flag, train_flag, gossip_lock,
                       gossip_params, gossip_device_buffer,
                       gossip_ps_weight, gossip_ps_factor):
        """ Gossip thread, which performs push-sum on model params """
        logger = make_logger(dist_config['rank'], dist_config['verbose'])

        if dist_config['comm_device'].type != 'cpu':
            gossip_stream = torch.cuda.Stream()
            dist._register_stream(gossip_stream)
        else:
            gossip_stream = torch.cuda.current_stream()

        # init gossip instance
        if dist_config['push_sum']:
            gossiper = PushSum(_flatten_tensors(gossip_params),
                               device=dist_config['comm_device'],
                               graph=dist_config['graph'],
                               mixing=dist_config['mixing'],
                               rank=dist_config['rank'],
                               world_size=dist_config['world_size'],
                               logger=logger)
        else:
            gossiper = PushPull(_flatten_tensors(gossip_params),
                                device=dist_config['comm_device'],
                                graph=dist_config['graph'],
                                rank=dist_config['rank'],
                                world_size=dist_config['world_size'],
                                mixing=dist_config['mixing'],
                                logger=logger)
        dist_config['graph'] = gossiper._graph_manager
        dist_config['mixing'] = gossiper._mixing_manager
        dist_config['gossiper'] = gossiper
        gossip_ps_factor[0] = gossiper.mixing_weights['lo']
        gossip_flag.set()

        # gossip loop
        while True:
            train_flag.wait()
            logger.debug('received train-flag')
            try:
                with torch.cuda.stream(gossip_stream):
                    # construct gossip tensor
                    out_msg = _flatten_tensors(gossip_params)
                    # gossip step
                    with gossip_lock:
                        in_msg, psw = gossiper.mix(out_msg,
                                                   gossip_ps_weight[0],
                                                   residual=True)
                        gossip_ps_factor[0] = gossiper.mixing_weights['lo']
                    # update gossip variables with residuals
                    gossip_ps_weight[0] = psw
                    for r, g in zip(_unflatten_tensors(in_msg, gossip_params),
                                    gossip_device_buffer):
                        g.copy_(r, non_blocking=True)
            except RuntimeError as e:
                logger.warning('received runtime error {}'.format(e))
                gossiper.clean_msg_buffers_()
                gossip_ps_weight[0] = -1
            finally:
                # give main thread go-ahead to read our gossip buffer
                train_flag.clear()
                gossip_flag.set()

    def __register_hooks(self):
        """
        Registers push-sum de-bias/bias hooks in pre-forward/post-backward
        passes in all leaf modules
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

            # convert model back to ps-numerator
            if self.distributed:
                self.ps_numerator()

        def queue_hook(*unused):
            Variable._execution_engine.queue_callback(hook)
        return queue_hook

    def __make_forward_pre_hook(self):
        self.logger.debug('making forward pre-hook')

        def hook(*unused):
            """ Query gossip queue and de-bias during forward pass """
            if self.distributed:
                # gossip during training (not inference)
                if self.gossip_enable:
                    non_blocking = self.num_updates < self.synch_freq
                    if self._query_gossip_queue(non_blocking):
                        self.num_updates = 0
                    else:
                        self.num_updates += 1
                    if self.overlap:
                        self.transfer_params()

                # convert model to de-biased estimate
                self.unbias()

        return hook

    def communicator_warmup(self):
        """ time the all-reducde code """
        if not self.distributed:
            return
        dist.barrier()
        time.sleep(5)
        dist.barrier()
