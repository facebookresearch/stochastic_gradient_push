# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import torch
from torch.nn import parallel

from fairseq.distributed_utils import c10d_status

from . import BaseFairseqModel


def DistributedFairseqModel(args, model):
    """
    Wrap a *model* to support distributed data parallel training.

    This is similar to the built-in DistributedDataParallel, but allows
    additional configuration of the DistributedDataParallel class to
    use, and also provides easier access to the wrapped model by
    forwarding requests for missing attributes to the wrapped model.

    Args:
        args (argparse.Namespace): fairseq args
        model (BaseFairseqModel): model to wrap
    """

    # determine which DDP class to extend
    assert isinstance(model, BaseFairseqModel)
    print (args.ddp_backend, args.dist_avg)
    if args.dist_process == 0:
        devices_lst=list(range(torch.cuda.device_count()))
    else:
        devices_lst=[args.device_id]
    print ('device ID: ', args.device_id, devices_lst)

    if args.ddp_backend == 'c10d':
        if c10d_status.is_default:
            ddp_class = parallel.DistributedDataParallel
        elif c10d_status.has_c10d:
            ddp_class = parallel._DistributedDataParallelC10d
        else:
            raise Exception(
                'Can\'t find c10d version of DistributedDataParallel. '
                'Please update PyTorch.'
            )
        init_kwargs = dict(
            module=model,
            device_ids=devices_lst, #[args.device_id],
            output_device=args.device_id,
            broadcast_buffers=False,
            bucket_cap_mb=args.bucket_cap_mb,
        )
    elif args.ddp_backend == 'no_c10d':
        if args.dist_avg == 'gossip':
            from gossip_module import GossipDataParallel
            ddp_class = GossipDataParallel
            cpu_comm = True if args.distributed_backend == 'tcp' else False
            comm_device = torch.device('cpu') if cpu_comm else torch.device('cuda')
            from gossip_module import DynamicDirectedExponentialGraph as DDEGraph
            init_kwargs = dict (module=model,
                                distributed=True,
                                graph=DDEGraph(args.distributed_rank, args.distributed_world_size, peers_per_itr=1),
                                comm_device=comm_device,
                                device_ids=devices_lst,
                                overlap=False,
                                synch_freq=0,
                                push_sum=True,
                                verbose=False)
        elif args.dist_avg == 'simple_gossip':
            from gossip_module import SimpleGossipDataParallel
            ddp_class = SimpleGossipDataParallel
            cpu_comm = True if args.distributed_backend == 'tcp' else False
            comm_device = torch.device('cpu') if cpu_comm else torch.device('cuda')
            from gossip_module import DynamicDirectedExponentialGraph as DDEGraph
            init_kwargs = dict (module=model,
                                distributed=True,
                                graph=DDEGraph(args.distributed_rank, args.distributed_world_size, peers_per_itr=1),
                                comm_device=comm_device,
                                device_ids=devices_lst,
                                output_device=args.device_id,
                                push_sum=True,
                                verbose=False)
        elif args.dist_avg == 'allreduce2':
            from gossip_module import AllReduceDataParallel
            ddp_class = AllReduceDataParallel
            cpu_comm = True if args.distributed_backend == 'tcp' else False
            comm_device = torch.device('cpu') if cpu_comm else torch.device('cuda')
            init_kwargs = dict(module=model,
                               device_ids=devices_lst,
                               distributed=True,
                               comm_device=comm_device,
                               verbose=False)
        else:
            if c10d_status.is_default:
                ddp_class = parallel.deprecated.DistributedDataParallel
            else:
                ddp_class = parallel.DistributedDataParallel
            init_kwargs = dict(
                module=model,
                #device_ids=[args.device_id],
                device_ids=devices_lst,
                output_device=args.device_id,
                broadcast_buffers=False,
            )
    else:
        raise ValueError('Unknown --ddp-backend: ' + args.ddp_backend)

    print (ddp_class)
    class _DistributedFairseqModel(ddp_class):
        """Extend DistributedDataParallel to check for missing
        attributes in the wrapped module."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def __getattr__(self, name):
            wrapped_module = super().__getattr__('module')
            if hasattr(wrapped_module, name):
                return getattr(wrapped_module, name)
            return super().__getattr__(name)

    return _DistributedFairseqModel(**init_kwargs)
