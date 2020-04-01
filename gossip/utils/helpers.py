# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
Collection of commonly used uitility functions
"""

import collections
import logging
import math
import sys

import torch
import torch.distributed as dist


def flatten_tensors(tensors):
    """
    Flatten dense tensors into a contiguous 1D buffer. Assume tensors are of
    same dense type.
    Since inputs are dense, the resulting tensor will be a concatenated 1D
    buffer. Element-wise operation on this buffer will be equivalent to
    operating individually.
    Arguments:
        tensors (Iterable[Tensor]): dense tensors to flatten.
    Returns:
        A 1D buffer containing input tensors.
    """
    if len(tensors) == 1:
        return tensors[0].view(-1).clone()
    flat = torch.cat([t.view(-1) for t in tensors], dim=0)
    return flat


def unflatten_tensors(flat, tensors):
    """
    View a flat buffer using the sizes of tensors. Assume that tensors are of
    same dense type, and that flat is given by flatten_dense_tensors.
    Arguments:
        flat (Tensor): flattened dense tensors to unflatten.
        tensors (Iterable[Tensor]): dense tensors whose sizes will be used to
            unflatten flat.
    Returns:
        Unflattened dense tensors with sizes same as tensors and values from
        flat.
    """
    outputs = []
    offset = 0
    for tensor in tensors:
        numel = tensor.numel()
        outputs.append(flat.narrow(0, offset, numel).view_as(tensor))
        offset += numel
    return tuple(outputs)


def group_by_dtype(tensors):
    """
    Returns a dict mapping from the tensor dtype to a list containing all
    tensors of that dtype.
    Arguments:
        tensors (Iterable[Tensor]): list of tensors.
    """
    tensors_by_dtype = collections.defaultdict(list)
    for tensor in tensors:
        tensors_by_dtype[tensor.dtype].append(tensor)
    return tensors_by_dtype


def communicate(tensors, communication_op):
    """
    Communicate a list of tensors.
    Arguments:
        tensors (Iterable[Tensor]): list of tensors.
        communication_op: a method or partial object which takes a tensor as
            input and communicates it. It can be a partial object around
            something like torch.distributed.all_reduce.
    """
    tensors_by_dtype = group_by_dtype(tensors)
    for dtype in tensors_by_dtype:
        flat_tensor = flatten_tensors(tensors_by_dtype[dtype])
        communication_op(tensor=flat_tensor)
        for f, t in zip(unflatten_tensors(flat_tensor, tensors_by_dtype[dtype]),
                        tensors_by_dtype[dtype]):
            t.set_(f)


def make_logger(rank, verbose=True):
    """
    Return a logger for writing to stdout;
    Arguments:
        rank (int): rank of node making logger
        verbose (bool): whether to set log-level to INFO; o.w. WARNING
    Returns:
        Python logger
    """
    logger = logging.getLogger(__name__)
    if not getattr(logger, 'handler_set', None):
        console = logging.StreamHandler(stream=sys.stdout)
        format_str = '{}'.format(rank)
        format_str += ': %(levelname)s -- %(threadName)s -- %(message)s'
        console.setFormatter(logging.Formatter(format_str))
        logger.addHandler(console)  # prints to console
        logger.handler_set = True
    if not getattr(logger, 'level_set', None):
        if verbose:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)
        logger.level_set = True
    return logger


def is_power_of(N, k):
    """
    Returns True if N is a power of k
    """
    assert isinstance(N, int) and isinstance(k, int)
    assert k >= 0 and N > 0
    if k == 0 and N == 1:
        return True
    if k in (0, 1) and N != 1:
        return False

    return k ** int(round(math.log(N, k))) == N


def create_process_group(ranks):
    """
    Creates and lazy intializes a new process group. Assumes init_process_group
    has already been called.
    Arguments:
        ranks (list<int>): ranks corresponding to the processes which should
            belong the created process group
    Returns:
        new process group
    """
    initializer_tensor = torch.Tensor([1])
    if torch.cuda.is_available():
        initializer_tensor = initializer_tensor.cuda()
    new_group = dist.new_group(ranks)
    dist.all_reduce(initializer_tensor, group=new_group)
    return new_group

def get_tcp_interface_name(network_interface_type='ethernet'):
    """
    Return the name of the ethernet interface which is up
    """
    network_interfaces = os.listdir('/sys/class/net')

    process = subprocess.Popen('ip link show up'.split(),
                               stdout=subprocess.PIPE)
    out, err = process.communicate()

    prefix_list_map = {
        'ethernet': ('ens', 'eth', 'enp'),
        'infiniband': ('ib'),
    }

    for network_interface in network_interfaces:
        prefix_list = prefix_list_map[network_interface_type]
        if (network_interface.startswith(prefix_list)
                and network_interface in out.decode('utf-8')):
            print('Using network interface {}'.format(network_interface))
            return network_interface
    print('List of network interfaces found:', network_interfaces)
    print('Prefix list being used to search:', prefix_list)
    raise Exception('No proper ethernet interface found')
