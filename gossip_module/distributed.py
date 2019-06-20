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

import functools
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
from .graph_manager import NPeerDynamicDirectedExponentialGraph as NPDDEGraph
from .mixing_manager import UniformMixing
from .utils import (
    create_process_group, communicate, flatten_tensors,
    group_by_dtype, make_logger, unflatten_tensors)

HEARTBEAT_TIMEOUT = 300  # maximum time to wait for message (seconds)


class GossipDataParallel(Module):
    """ Distributed Gossip model wrapper """

    def __init__(self, module, device_ids=None, rank=None, world_size=None,
                 graph=None, mixing=None, comm_device=None, push_sum=True,
                 overlap=False, synch_freq=0, verbose=False, use_streams=True,
                 nprocs_per_node=1, local_node_group=None):
        super(GossipDataParallel, self).__init__()

        # devices available locally
        if device_ids is None:
            device_ids = list(range(torch.cuda.device_count()))
        self.output_device = device_ids[0]
        self.device_ids = device_ids

        self.nprocs_per_node = nprocs_per_node

        if world_size is None or rank is None:
            assert dist.is_initialized()
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        self.process_rank = rank

        if self.nprocs_per_node > 1:
            self.local_rank = self.process_rank % self.nprocs_per_node
            world_size //= nprocs_per_node
            rank //= nprocs_per_node
            if local_node_group is None:
                for node in range(world_size):
                    node_processes_ranks = list(
                        range(node * self.nprocs_per_node,
                              (node + 1) * self.nprocs_per_node))
                    # Process group to communicate between processes on this
                    # machine
                    new_local_group = create_process_group(
                        node_processes_ranks)
                    if self.process_rank in node_processes_ranks:
                        self.local_node_group = new_local_group
            else:
                self.local_node_group = local_node_group
        else:
            self.local_rank = 0

        # put model on output device
        self.module = module
        first_param_dtype = next(self.module.parameters()).dtype

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

        # choose communication device based on backend
        if comm_device is None:
            cpu_comm = True if dist.get_backend() == 'gloo' else False
            comm_device = torch.device('cpu') if cpu_comm else torch.device('cuda')
        self.__cpu_comm = comm_device.type == 'cpu'

        if graph is None:
            graph = NPDDEGraph(
                rank, world_size, self.nprocs_per_node, self.local_rank)

        if mixing is None:
            mixing = UniformMixing(graph, comm_device)

        # distributed backend config
        self.dist_config = {
            'verbose': verbose,
            'comm_device': comm_device,
            'graph': graph,
            'mixing': mixing,
            'push_sum': push_sum,
            'rank': rank,
            'process_rank': self.process_rank,
            'world_size': world_size,
            'cpu_comm': self.__cpu_comm
        }
        self.overlap = overlap
        self.synch_freq = synch_freq
        self.num_updates = 0
        self.asynch = synch_freq > 0

        # logger used to print to stdout
        self.logger = make_logger(rank, verbose)

        # push-sum weight=1.0 ==> distributed averaging
        self.ps_weight = torch.ones(1, device=comm_device).type(
            first_param_dtype)
        self.nprocs_per_node_device = torch.tensor(
            [self.nprocs_per_node], device=comm_device,
            dtype=first_param_dtype)
        self.is_ps_numerator = False

        # prepare parameters for gossip
        self.gossip_enable = True
        self.gossiping = False
        self.params_mixed = True
        self.gossip_ps_factor = torch.zeros(1, device=comm_device).type(
            first_param_dtype)
        self.gossip_ps_weight = self.ps_weight.clone()
        self.gossip_params = []
        self.gossip_device_buffer = []
        for p in module.parameters():
            cp = p.clone().detach_()
            cp = cp.cpu().pin_memory() if self.__cpu_comm else cp.cuda()
            self.gossip_params.append(cp)
            self.gossip_device_buffer.append(cp)

        # prepare gossip process control objects
        self.gossip_lock = threading.Lock()
        self.gossip_flag = threading.Event()
        self.train_flag = threading.Event()

        if self.dist_config['comm_device'].type != 'cpu' and use_streams:
            self.gossip_stream = torch.cuda.Stream()
        else:
            self.gossip_stream = torch.cuda.current_stream()

        if self.process_rank % self.nprocs_per_node == 0:
            self.gossip_thread = threading.Thread(
                target=GossipDataParallel._gossip_target,
                args=(self.dist_config,
                      self.gossip_flag,
                      self.train_flag,
                      self.gossip_lock,
                      self.gossip_params,
                      self.gossip_device_buffer,
                      self.gossip_ps_weight,
                      self.gossip_ps_factor,
                      self.gossip_stream))
            self.gossip_thread.daemon = True
            self.gossip_thread.name = 'Gossip-Thread'
            self.gossip_thread.start()
        else:
            self.gossip_flag.set()
        # wait for thread to complete initialization
        self.gossip_flag.wait()
        self.gossip_flag.clear()
        # lazy mixing avoids additional bias/de-bias steps
        self.lazy_mixing = (
            not self.asynch and self.dist_config['mixing'].is_regular() and
            not self.overlap)
        self.lazy_ps_factor = self.gossip_ps_factor.clone()
        self.logger.debug('lazy mixing: {}'.format(self.lazy_mixing))

        # register ps/grad-reduction hooks
        self.__register_hooks()

    def update_gossiper(self, attr, val):
        self.logger.debug('waiting for gossip lock')
        with self.gossip_lock:
            self.logger.debug('gossip lock received')
            for gossiper in self.dist_config['gossipers'].values():
                if val == getattr(gossiper, attr):
                    self.logger.debug('nothing to update')
                    return
                # update attr
                self.logger.debug('setting gossiper {} to {}'.format(attr, val))
                setattr(gossiper, attr, val)

    def state_dict(self, finish_gossip=True):
        # If user is saving the model, complete the gossip to avoid losing
        # the information which has been sent by a peer. If _query_gossip_queue
        # is not called here, it would only be called in the next
        # pre_forward_hook and information sent by the peer will be lost
        # if the checkpoint is restored
        if finish_gossip:
            self._query_gossip_queue()

        super_dict = super(GossipDataParallel, self).state_dict()
        supplanted_dict = {'state_dict': super_dict,
                           'ps_weight': self.ps_weight.cpu(),
                           'is_ps_numerator': self.is_ps_numerator}
        return supplanted_dict

    def load_state_dict(self, load_dict):
        state_dict = load_dict['state_dict']
        super(GossipDataParallel, self).load_state_dict(state_dict)
        self.ps_weight = load_dict['ps_weight'].to(
            device=self.dist_config['comm_device'])
        self.is_ps_numerator = load_dict['is_ps_numerator']

    def forward(self, *inputs, **kwargs):
        """ Forward pass performed in parallel across all devices on node """
        # scatter inputs onto devices
        inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids)
        if self.nprocs_per_node > 1:
            self._sync_params_multiprocess()
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
        """ Synchronize parameters across devices (intra-node) """
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
        buffers = [b.data for b in self.module.buffers()]
        if len(buffers) > 0:
            result = broadcast_coalesced(buffers, self.device_ids,
                                         self.broadcast_bucket_size)
            for tensors, module in zip(result[1:], self._module_copies[1:]):
                for tensor, buf in zip(tensors, module.buffers()):
                    buf.data.set_(tensor)

    def _sync_params_multiprocess(self):
        """ Synchronize parameters across devices (intra-node) """
        # intra-node parameter sync
        params = [p.data for p in self.module.parameters()]
        communication_op = functools.partial(
            dist.broadcast,
            src=(self.dist_config['rank'] * self.nprocs_per_node),
            group=self.local_node_group)
        communicate(params, communication_op)

        # intra-node buffer sync
        buffers = [b.data for b in self.module.buffers()]
        if len(buffers) > 0:
            buffers = [b.data for b in self.module.buffers()]
            communication_op = functools.partial(
                dist.broadcast,
                src=(self.dist_config['rank'] * self.nprocs_per_node),
                group=self.local_node_group)
            communicate(buffers, communication_op)

    def ps_numerator(self):
        """ Convert model params to ps-numerator """
        if not self.is_ps_numerator:
            ps_weight = self.ps_weight
            if not self.lazy_mixing:
                for p in self.module.parameters():
                    p.data.mul_(ps_weight.type(p.data.dtype))
            self.is_ps_numerator = True

    def unbias(self):
        """ Convert moel params to de-biased estimate """
        if self.is_ps_numerator:
            ps_weight = self.ps_weight
            if not self.lazy_mixing:
                for p in self.module.parameters():
                    p.data.div_(ps_weight.type(p.data.dtype))
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
        self._query_gossip_queue(non_blocking=self.asynch)

    def block(self):
        self.logger.info('blocking')
        dist.barrier()

    def sync_comms(self):
        self._query_gossip_queue(non_blocking=False)

    def _query_gossip_queue(self, non_blocking=False):
        """ Check gossip-queue for push-sum residuals and update model """
        if not self.gossip_enable:
            return

        self.logger.debug('querying gossip queue')

        # no gossip happening right now so just return
        if not self.gossiping:
            if self.process_rank % self.nprocs_per_node == 0:
                self.logger.warning('not gossiping right now')
            return False

        if not non_blocking:
            if not self.gossip_flag.wait(timeout=HEARTBEAT_TIMEOUT):
                raise NameError('Gossip flag timeout')
                sys.exit()  # HEARTBEAT monitor

        # query gossip thread
        if self.gossip_flag.is_set():
            self.logger.debug('received gossip flag')

            # atomic gossip was interrupted so try again
            if self.gossip_ps_weight[0] == -1:
                self.gossip_flag.clear()
                self.params_mixed = True
                self.gossiping = False
                self.transfer_params(mix=False)
                return False

            self.lazy_ps_factor.copy_(self.gossip_ps_factor)

            # convert model-params to ps numerators b4 adding residuals
            self.ps_numerator()

            # add residuals
            self.ps_weight += self.gossip_ps_weight
            if self.lazy_mixing:
                self.ps_weight *= self.lazy_ps_factor
            for p, r in zip(self.module.parameters(),
                            self.gossip_device_buffer):
                p.data.add_(r)
                if self.lazy_mixing:
                    p.data.mul_(self.lazy_ps_factor.type(p.data.dtype))

            # update flags
            self.logger.debug('updated ps-weight {}'.format(self.ps_weight))
            self.logger.debug('updated model params')
            self.gossip_flag.clear()
            self.params_mixed = True
            self.gossiping = False
            return True

    def transfer_params(self, mix=True):
        """ Transfers COPY of model parameters to gossip queue """
        if (not self.gossip_enable or
                self.process_rank % self.nprocs_per_node != 0):
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
            self.ps_weight *= self.gossip_ps_factor
        self.gossip_ps_weight.copy_(self.ps_weight)
        # --
        # params gpu-gpu copy (fast)
        # --
        for p, gossip_device_buffer_elem in zip(
                self.module.parameters(), self.gossip_device_buffer):
            if mix:
                p.data.mul_(self.gossip_ps_factor.type(p.data.dtype))
            gossip_device_buffer_elem.data.copy_(p)
        # --
        # buffer to gossip-thread copy (potentially slow, but asynchronous)
        # --
        self.gossip_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.gossip_stream):
            for b, gp in zip(self.gossip_device_buffer, self.gossip_params):
                gp.copy_(b, non_blocking=True)

        # --

        # update flags
        self.logger.debug('transfered model params')
        self.params_mixed = False
        self.gossiping = True
        self.train_flag.set()
        return True

    @staticmethod
    def _gossip_into_receive_buffer(send_buffer, gossiper, receive_buffer,
                                    gossip_ps_weight, gossip_lock,
                                    dist_config):
        # flatten parameters before sending
        out_msg = flatten_tensors(send_buffer)

        # send and receive parameters
        with gossip_lock:
            in_msg, ps_weight = gossiper.mix(out_msg, gossip_ps_weight,
                                             residual=True)
            ps_factor = gossiper.mixing_weights['lo']

        # unflatten parameters
        for r, g in zip(unflatten_tensors(in_msg, send_buffer),
                        receive_buffer):
            if dist_config['cpu_comm']:
                g.copy_(r, non_blocking=True)
            else:
                g.data.copy_(r)

        return ps_weight, ps_factor

    @staticmethod
    def _gossip_target(dist_config, gossip_flag, train_flag, gossip_lock,
                       gossip_params, gossip_device_buffer,
                       gossip_ps_weight, gossip_ps_factor, gossip_stream):
        """ Gossip thread, which performs push-sum on model params """
        logger = make_logger(dist_config['rank'], dist_config['verbose'])

        gossip_params_by_dtype = group_by_dtype(gossip_params)
        gossip_device_buffer_by_dtype = group_by_dtype(gossip_device_buffer)

        gossipers = {}
        # init gossip instance
        gossiper_class = PushSum if dist_config['push_sum'] else PushPull
        for dtype in gossip_params_by_dtype:
            gossipers[dtype] = gossiper_class(
                flatten_tensors(gossip_params_by_dtype[dtype]),
                device=dist_config['comm_device'],
                graph=dist_config['graph'],
                mixing=dist_config['mixing'],
                rank=dist_config['process_rank'],
                world_size=dist_config['world_size'],
                logger=logger)

        dist_config['gossipers'] = gossipers
        gossip_ps_factor.data.copy_(
            gossipers[list(gossipers)[0]].mixing_weights['lo'])
        gossip_flag.set()

        # gossip loop
        while True:
            train_flag.wait()
            logger.debug('received train-flag')
            try:
                with torch.cuda.stream(gossip_stream):
                    for dtype in gossip_params_by_dtype:
                        ps_weight, ps_factor = GossipDataParallel._gossip_into_receive_buffer(
                            gossip_params_by_dtype[dtype], gossipers[dtype],
                            gossip_device_buffer_by_dtype[dtype],
                            gossip_ps_weight, gossip_lock, dist_config)
                    gossip_ps_weight.copy_(ps_weight)
                    gossip_ps_factor.copy_(ps_factor)
            except RuntimeError as e:
                logger.warning('received runtime error {}'.format(e))
                for gossiper in gossipers.values():
                    gossiper.clean_msg_buffers_()
                gossip_ps_weight.fill_(-1)
            finally:
                # Make sure all queued operations are complete
                gossip_stream.synchronize()
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

            if self.nprocs_per_node > 1:
                grads = []
                for p in self.module.parameters():
                    if not p.requires_grad or p.grad is None:
                        continue
                    p.grad.data.div_(self.nprocs_per_node_device.type(
                        p.grad.data.dtype))
                    grads.append(p.grad.data)

                communication_op = functools.partial(
                    dist.all_reduce, group=self.local_node_group)
                communicate(grads, communication_op)

            # convert model back to ps-numerator
            self.ps_numerator()

        def queue_hook(*unused):
            Variable._execution_engine.queue_callback(hook)
        return queue_hook

    def __make_forward_pre_hook(self):
        self.logger.debug('making forward pre-hook')

        def hook(*unused):
            """ Query gossip queue and de-bias during forward pass """
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
