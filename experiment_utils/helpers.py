# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
Collection of commonly used utility functions
"""

import logging
import sys

import torch


def _flatten_tensors(tensors):
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


def _unflatten_tensors(flat, tensors):
    """
    View a flat buffer using the sizes of tensors. Assume that tensors are of
    same dense type, and that flat is given by _flatten_dense_tensors.
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


def make_logger(rank, verbose=True):
    """
    Return a logger for writing to stdout; only one logger for each application
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
