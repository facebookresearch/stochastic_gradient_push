# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import os
import socket
import subprocess
import torch

from train import main as single_process_main
from fairseq import distributed_utils, options


def main(args):
    if args.distributed_init_method is None and args.distributed_port > 0:
        # We can determine the init method automatically for Slurm.
        node_list = os.environ.get('SLURM_JOB_NODELIST')
        if node_list is not None:
            try:
                if args.distributed_backend == 'nccl' and args.dist_avg == 'allreduce':
                    hostnames = subprocess.check_output(['scontrol', 'show', 'hostnames', node_list])
                    args.distributed_init_method = 'tcp://{host}:{port}'.format(
                        host=hostnames.split()[0].decode('utf-8'),
                        port=args.distributed_port)
                    args.distributed_rank = int(os.environ.get('SLURM_PROCID'))
                    args.device_id = int(os.environ.get('SLURM_LOCALID')) % torch.cuda.device_count()
                else:
                    args.master_addr = os.environ['HOSTNAME']
                    if args.distributed_backend == 'mpi':
                        if args.dist_process:
                            args.distributed_rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
                            #print (int(os.environ['OMPI_COMM_WORLD_RANK']), int(os.environ['OMPI_COMM_WORLD_NODE_RANK']), os.environ['OMPI_COMM_WORLD_LOCAL_RANK'], int(os.environ['OMPI_UNIVERSE_SIZE']))
                            #args.distributed_rank = int(os.environ['OMPI_COMM_WORLD_NODE_RANK'])
                            args.device_id = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK']) % torch.cuda.device_count()
                        else:
                            args.distributed_rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
                            args.distributed_world_size = int(os.environ['OMPI_UNIVERSE_SIZE'])
                            args.device_id = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
                    else:
                        if args.dist_process:
                            args.distributed_rank = int(os.environ.get('SLURM_PROCID'))
                            args.device_id = int(os.environ.get('SLURM_LOCALID')) % torch.cuda.device_count()
                        else:
                            args.distributed_rank = int(os.environ['SLURM_PROCID'])
                            args.distributed_world_size = int(os.environ['SLURM_NTASKS'])
                            args.device_id = int(os.environ['SLURM_LOCALID'])
                    args.distributed_init_method = 'tcp://{host}:{port}'.format(
                        host=args.master_addr,
                        port=args.distributed_port)


            except subprocess.CalledProcessError as e:  # scontrol failed
                raise e
            except FileNotFoundError as e:  # Slurm is not installed
                pass
    if args.distributed_init_method is None and args.distributed_port is None:
        raise ValueError('--distributed-init-method or --distributed-port '
                         'must be specified for distributed training')


    args.distributed_rank = distributed_utils.distributed_init(args)
    print('| initialized host {} as rank {}'.format(socket.gethostname(), args.distributed_rank))
    print ('init:', args.distributed_rank, args.distributed_world_size, args.distributed_init_method)

    single_process_main(args)


if __name__ == '__main__':
    parser = options.get_training_parser()
    args = options.parse_args_and_arch(parser)
    main(args)
