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
import os
import subprocess
import sys


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
