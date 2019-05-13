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

from .helpers import make_logger


class ClusterManager(object):
    """ Class keeps track of external SLURM cluster signals """

    USER_NAME_IS_SET = False
    MASTER_RANK = 0

    @staticmethod
    def set_user_name(user_name):
        ClusterManager.USER_NAME_IS_SET = True
        ClusterManager.USER_NAME = user_name
        ClusterManager.CHECKPOINT_DIR = 'FIXME'

    def __init__(self, rank, world_size, bs_fname, state,
                 model_tag='', callback=None, all_workers=False):
        """
        Constructor for ClusterManager()

        :param rank: The agent's rank (unique Id)
        :param world_size: Number of agents used for training
        :param bs_fname: The filename of the batch script launching the job
        :param state: Dictionary used to encode training state
        :param all_workers: Whether to save all workers' models in checkpoints
        :param callback: function to execute when SIGTERM is received
        """
        assert ClusterManager.USER_NAME_IS_SET

        self.rank = rank
        self.world_size = world_size
        self.bs_fname = bs_fname
        self.state = state
        self.all_workers = all_workers
        self.main_pid = os.getpid()
        self.signal_received = False
        self.halt = False
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

    def save_checkpoint(self, epoch_id=None):
        if not self.signal_received:
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

        self.save_checkpoint()

        self.signal_received = True

        # if 'halt' flag is set, exit without relaunching another job
        if self.halt:
            self.logger.info('Job is done, exiting')
            # TODO: add some synchronization barrier and then call exit()
            # For now; hang tight until job is terminated (exiting early might
            # result in unexpected behaviour)
            while True:
                pass

        # relaunch job on cluster starting from checkpoint
        if self.rank == 0 and os.getpid() == self.main_pid:
            """ Only launch job from the main process of the rank 0 agent """

            command = 'sbatch --begin=now+120 ' + self.bs_fname + ' True'
            self.logger.info('Relaunching: ' + command)
            if os.system(command):
                raise RuntimeError('sbatch failed')
            self.logger.info('New job submitted to the queue')

        self.logger.info('Waiting until termination')
