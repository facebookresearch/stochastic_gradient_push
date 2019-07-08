# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
Cluster manager providing model checkpointing services in case of job
preemption or termination

"""

import os
import signal
import shutil

import torch
import torch.distributed as dist

from .helpers import make_logger


class ClusterManager(object):
    """ Class keeps track of external SLURM cluster signals. If training with
    multiple processes, this assumes dist.init_process_group has been called
    prior to calling the constructor """

    MASTER_RANK = 0
    CHECKPOINT_DIR = None

    @staticmethod
    def set_checkpoint_dir(checkpoint_dir):
        ClusterManager.CHECKPOINT_DIR = checkpoint_dir

    def __init__(self, rank, world_size, state,
                 model_tag='', callback=None, all_workers=False):
        """
        Constructor for ClusterManager()

        :param rank: The agent's rank (unique Id)
        :param world_size: Number of agents used for training
        :param state: Dictionary used to encode training state
        :param model_tag: Tag used in the name of the checkpoint file
        :param callback: function to execute when SIGTERM is received
        :param all_workers: Whether to save all workers' models in checkpoints
        """
        assert ClusterManager.CHECKPOINT_DIR is not None

        self.rank = rank
        self.world_size = world_size
        self.state = state
        self.all_workers = all_workers
        self.main_pid = os.getpid()
        self.signal_tensor = torch.zeros(1)
        if torch.cuda.is_available():
            self.signal_tensor = self.signal_tensor.cuda()
        self.signal_handlers_installed = False
        self.logger = make_logger(rank)
        self.callback = None

        if all_workers:
            model_rank = rank
        else:
            model_rank = ClusterManager.MASTER_RANK

        self.model_tag = model_tag

        self.checkpoint_fname = 'checkpoint_r' + str(model_rank) + \
                                '_n' + str(world_size) + \
                                '.pth.tar'
        self.model_best_fname = 'model_best_r' + str(model_rank) + \
                                '_n' + str(world_size) + \
                                '.pth.tar'
        self.checkpoint_fpath = ClusterManager.CHECKPOINT_DIR \
            + self.model_tag + self.checkpoint_fname
        self.model_best_fpath = ClusterManager.CHECKPOINT_DIR \
            + self.model_tag + self.model_best_fname

        self.install_signal_handlers()

        if self.world_size > 1:
            assert dist.is_initialized()
            self.process_group = dist.new_group(list(range(self.world_size)))

    def save_checkpoint(self, epoch_id=None, requeue_on_signal=True):
        # To find out if a signal is received in any process
        if requeue_on_signal and self.world_size > 1:
            dist.all_reduce(self.signal_tensor, group=self.process_group)

        self.logger.info('Saving checkpoint')
        if self.all_workers or self.rank == ClusterManager.MASTER_RANK:
            if epoch_id is None:
                checkpoint_fpath = self.checkpoint_fpath
            else:
                checkpoint_fpath = ClusterManager.CHECKPOINT_DIR \
                    + 'ep' + str(epoch_id) + '_' \
                    + self.model_tag + self.checkpoint_fname
            torch.save(self.state, checkpoint_fpath)
            if self.state['is_best']:
                shutil.copyfile(checkpoint_fpath,
                                self.model_best_fpath)
                self.state['is_best'] = False

        if requeue_on_signal and self.signal_tensor[0] > 0:
            self.logger.info('Atleast 1 process received SIGUSR1. Terminating')

            # relaunch job on cluster starting from checkpoint only for
            # main process of the rank 0 agent
            if self.rank == 0 and os.getpid() == self.main_pid:
                command = f'scontrol requeue {os.environ["SLURM_JOB_ID"]}'
                self.logger.info('Relaunching: ' + command)
                if os.system(command):
                    raise RuntimeError('sbatch failed')
                self.logger.info('New job submitted to the queue')

            self.logger.info('Terminating')
            sys.exit(0)

    def install_signal_handlers(self):
        self.logger.info('Signal handlers installed')
        signal.signal(signal.SIGUSR1, self.SIGUSR1Handler)
        signal.signal(signal.SIGTERM, self.SIGTERMHandler)
        self.signal_handlers_installed = True

    def SIGTERMHandler(self, signum, frame):
        """
        Ignore SIGTERM preemption signal (doesn't stop preemption);
        instead SIGUSR1 will be moved up and handled accordingly
        """
        self.logger.info('Received SIGTERM')

    def SIGUSR1Handler(self, signum, frame):
        """ Cleanup and relaunch job following SIGUSR1 signal """

        self.logger.info('Received SIGUSR1')

        if self.callback is not None:
            self.callback()

        self.signal_tensor[0] = 1
