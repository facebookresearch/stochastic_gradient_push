# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
Gossip SGD

Distributed data parallel training of a ResNet-50 on ImageNet using
Asynchronous Distributed Parallel Stochastic Gradient Descent

Derived from https://github.com/pytorch/examples/blob/master/imagenet/main.py
"""

import argparse
import copy
import os
import socket
import time

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms

from torchvision.models.resnet import Bottleneck
from torch.nn.parameter import Parameter

from experiment_utils import make_logger
from experiment_utils import Meter
from experiment_utils import ClusterManager
from gossip_module import AllReduceDataParallel
from gossip_module import GossipDataParallel, SimpleGossipDataParallel
from gossip_module import BilatGossipDataParallel
from gossip_module import DynamicDirectedExponentialGraph as DDEGraph
from gossip_module import DynamicBipartiteExponentialGraph as DBEGraph
from gossip_module import DynamicDirectedLinearGraph as DDLGraph
from gossip_module import DynamicBipartiteLinearGraph as DBLGraph
from gossip_module import UniformMixing

GRAPH_TOPOLOGIES = {
    0: DDEGraph,  # Dynamic Directed Exponential
    1: DBEGraph,  # Dynamic Bipartite Exponential
    2: DDLGraph,  # Dynamic Directed Linear
    3: DBLGraph  # Dynamic Bipartite Linear
}

MIXING_STRATEGIES = {
    0: UniformMixing  # assign weights uniformly
}

# path to train/validation data
TRAIN_DIR = 'PATH_to_IMAGENET/train'
VAL_DIR = 'PATH_to_IMAGENET/val'

# --------------------------------------------------------------------------- #
# Parse command line arguments (CLAs):
# --------------------------------------------------------------------------- #
parser = argparse.ArgumentParser(description='Playground')
parser.add_argument('--all_reduce', default='False', type=str,
                    help='whether to use all-reduce or gossip')
parser.add_argument('--bilat', default='False', type=str,
                    help='whether to use bi-lat gossip')
parser.add_argument('--shared_fpath', default='', type=str,
                    help='file path to use for global iteration tracking')
parser.add_argument('--batch_size', default=32, type=int,
                    help='per-agent batch size')
parser.add_argument('--single_threaded', default='False', type=str,
                    help='whether to use single-threaded model wrapper')
parser.add_argument('--distributed', default='True', type=str,
                    help='whether to run script in distributed mode')
parser.add_argument('--lr', default=0.1, type=float,
                    help='reference learning rate (for 256 sample batch-size)')
parser.add_argument('--num_dataloader_workers', default=10, type=int,
                    help='number of dataloader workers to fork from main')
parser.add_argument('--num_epochs', default=90, type=int,
                    help='number of epochs to train')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='optimization momentum')
parser.add_argument('--weight_decay', default=1e-4, type=float,
                    help='regularization applied to non batch-norm weights')
parser.add_argument('--nesterov', default='False', type=str,
                    help='whether to use nesterov style momentum'
                         'otherwise will use regular Polyak momentum')
parser.add_argument('--push_sum', default='True', type=str,
                    help='whether to use push-sum or push-pull gossip')
parser.add_argument('--graph_type', default=0, type=int,
                    help='the graph topology to use for gossip'
                         'cf. the gossip_module graph_manager for available'
                         'graph topologies and their corresponding int-id')
parser.add_argument('--mixing_strategy', default=0, type=int,
                    help='the mixing strategy to use for gossip'
                         'cf. the gossip_module mixing_manager for available'
                         'mixing strategies and their corresponding int-id.')
parser.add_argument('--schedule', nargs='+', type=float,
                    help='learning rate schedule')
parser.add_argument('--peers_per_itr_schedule', nargs='+', type=int,
                    help='epoch schedule of num peers to send msgs to;'
                         'the expected format is list[epoch, num_peers]'
                         'if manually specifying peers_per_itr_schedule,'
                         'you must specify num_peers at epoch 0; i.e.,'
                         'list must contain: 0, __num_peers_at_epoch_0__')
parser.add_argument('--overlap', default='False', type=str,
                    help='whether to overlap communication with computation')
parser.add_argument('--synch_freq', default=0, type=int,
                    help='max number of iterations to go without synchronizing'
                         'communication between nodes')
parser.add_argument('--warmup', default='False', type=str,
                    help='whether to warmup learning rate for first 5 epochs')
parser.add_argument('--seed', default=47, type=int,
                    help='seed used for ALL stochastic elements in script')
parser.add_argument('--resume', default='False', type=str,
                    help='whether to resume from previously saved checkpoint')
parser.add_argument('--backend', default='nccl', type=str,
                    help='torch.distributed backend')
parser.add_argument('--bs_fpath', default='', type=str,
                    help='batch-script file path to resubmit on preemption')
parser.add_argument('--tag', default='', type=str,
                    help='tag used to prepend checkpoint file names')
parser.add_argument('--print_freq', default=10, type=int,
                    help='frequency (itr.) with which to print train stats')
parser.add_argument('--verbose', default='True', type=str,
                    help='whether to log everything or just warnings/errors')
parser.add_argument('--train_fast', default='False', type=str,
                    help='whether to run script with only one validation run'
                         '(at the end once the model is trained)')
parser.add_argument('--checkpoint_all', default='True', type=str,
                    help='True: save each agents model at each itr'
                         'False: save just one (rank 0) model at each itr')
parser.add_argument('--master_port', default='40100', type=str,
                    help='port used to initialize distributed backend')
parser.add_argument('--user_name', default='user', type=str,
                    help='user-name used to define directory for log-files')
# --------------------------------------------------------------------------- #


def main():

    global args, state, log
    args = parse_args()

    log = make_logger(args.rank, args.verbose)
    log.info('args: {}'.format(args))
    log.info(socket.gethostname())

    # seed for reproducibility
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    if args.distributed and not args.bilat:
        # initialize torch distributed backend
        os.environ['MASTER_ADDR'] = args.master_addr
        os.environ['MASTER_PORT'] = args.master_port
        dist.init_process_group(backend=args.backend,
                                world_size=args.world_size,
                                rank=args.rank)
        log.info('my dist rank {}'.format(dist.get_rank()))

    # init model, loss, and optimizer
    model = init_model()
    if args.all_reduce:
        model = AllReduceDataParallel(model,
                                      distributed=args.distributed,
                                      comm_device=args.comm_device,
                                      verbose=args.verbose)
    elif args.bilat:
        model = BilatGossipDataParallel(model,
                                        distributed=args.distributed,
                                        master_addr=args.master_addr,
                                        master_port=args.master_port,
                                        backend=args.backend,
                                        world_size=args.world_size,
                                        rank=args.rank,
                                        graph=args.graph,
                                        mixing=args.mixing,
                                        comm_device=args.comm_device,
                                        lr=args.lr,
                                        momentum=args.momentum,
                                        weight_decay=args.weight_decay,
                                        nesterov=args.nesterov,
                                        verbose=args.verbose)
    else:
        if args.single_threaded:
            model = SimpleGossipDataParallel(model,
                                             distributed=args.distributed,
                                             graph=args.graph,
                                             comm_device=args.comm_device,
                                             push_sum=args.push_sum,
                                             verbose=args.verbose)
        else:
            model = GossipDataParallel(model,
                                       distributed=args.distributed,
                                       graph=args.graph,
                                       mixing=args.mixing,
                                       comm_device=args.comm_device,
                                       push_sum=args.push_sum,
                                       overlap=args.overlap,
                                       synch_freq=args.synch_freq,
                                       verbose=args.verbose)
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=args.nesterov)
    optimizer.zero_grad()

    # dictionary used to encode training state
    state = {}
    update_state(state, {
            'epoch': 0, 'itr': 0, 'best_prec1': 0, 'is_best': True,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'elapsed_time': 0,
            'batch_meter': Meter(ptag='Time').__dict__,
            'data_meter': Meter(ptag='Data').__dict__,
            'nn_meter': Meter(ptag='Forward/Backward').__dict__
    })

    # module used to relaunch jobs and handle external termination signals
    cmanager = ClusterManager(rank=args.rank,
                              world_size=args.world_size,
                              bs_fname=args.bs_fpath,
                              model_tag=args.tag,
                              state=state,
                              all_workers=args.checkpoint_all)

    # resume from checkpoint
    if args.resume:
        if os.path.isfile(cmanager.checkpoint_fpath):
            log.info("=> loading checkpoint '{}'"
                     .format(cmanager.checkpoint_fpath))
            checkpoint = torch.load(cmanager.checkpoint_fpath)
            update_state(state, {
                          'epoch': checkpoint['epoch'],
                          'itr': checkpoint['itr'],
                          'best_prec1': checkpoint['best_prec1'],
                          'is_best': False,
                          'state_dict': checkpoint['state_dict'],
                          'optimizer': checkpoint['optimizer'],
                          'elapsed_time': checkpoint['elapsed_time'],
                          'batch_meter': checkpoint['batch_meter'],
                          'data_meter': checkpoint['data_meter'],
                          'nn_meter': checkpoint['nn_meter']
            })
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            log.info("=> loaded checkpoint '{}' (epoch {}; itr {})"
                     .format(cmanager.checkpoint_fpath,
                             checkpoint['epoch'], checkpoint['itr']))
        else:
            log.info("=> no checkpoint found at '{}'"
                     .format(cmanager.checkpoint_fpath))

    # enable low-level optimization of compute graph using cuDNN library?
    cudnn.benchmark = True

    # meters used to compute timing stats
    batch_meter = Meter(state['batch_meter'])
    data_meter = Meter(state['data_meter'])
    nn_meter = Meter(state['nn_meter'])

    # initalize log file
    if not args.resume:
        with open(args.out_fname, 'w') as f:
            print('BEGIN-TRAINING\n'
                  'World-Size,{ws}\n'
                  'Num-DLWorkers,{nw}\n'
                  'Batch-Size,{bs}\n'
                  'Epoch,itr,BT(s),avg:BT(s),std:BT(s),'
                  'NT(s),avg:NT(s),std:NT(s),'
                  'DT(s),avg:DT(s),std:DT(s),'
                  'Loss,avg:Loss,Prec@1,avg:Prec@1,Prec@5,avg:Prec@5,val'
                  .format(ws=args.world_size,
                          nw=args.num_dataloader_workers,
                          bs=args.batch_size), file=f)

    # create distributed data loaders
    loader, sampler = make_dataloader(args, train=True)
    if not args.train_fast:
        val_loader = make_dataloader(args, train=False)

    # start all agents' training loop at same time
    model.block()
    start_itr = state['itr']
    start_epoch = state['epoch']
    elapsed_time = state['elapsed_time']
    begin_time = time.time() - state['elapsed_time']
    epoch = start_epoch
    stopping_criterion = epoch >= args.num_epochs
    while not stopping_criterion:

        # deterministic seed used to load agent's subset of data
        sampler.set_epoch(epoch + args.seed * 90)

        if not args.all_reduce and not args.bilat:
            # update the model's peers_per_itr attribute
            update_peers_per_itr(model, epoch)

        train(model, criterion, optimizer,
              batch_meter, data_meter, nn_meter,
              loader, epoch, start_itr, begin_time)

        start_itr = 0
        if not args.train_fast:
            # update state after each epoch
            elapsed_time = time.time() - begin_time
            update_state(state, {
                'epoch': epoch + 1, 'itr': start_itr,
                'is_best': False,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'elapsed_time': elapsed_time,
                'batch_meter': batch_meter.__dict__,
                'data_meter': data_meter.__dict__,
                'nn_meter': nn_meter.__dict__
            })
            # evaluate on validation set and save checkpoint
            prec1 = validate(val_loader, model, criterion)
            with open(args.out_fname, '+a') as f:
                print('{ep},{itr},{bt},{nt},{dt},'
                      '{filler},{filler},'
                      '{filler},{filler},'
                      '{filler},{filler},'
                      '{val}'
                      .format(ep=epoch, itr=-1,
                              bt=batch_meter,
                              dt=data_meter, nt=nn_meter,
                              filler=-1, val=prec1), file=f)
            cmanager.save_checkpoint()
            # sycnhronize models at the end of validation run
            model.block()

        epoch += 1
        if args.bilat:
            stopping_criterion = args.global_epoch >= args.num_epochs
        else:
            stopping_criterion = epoch >= args.num_epochs

    if args.train_fast:
        val_loader = make_dataloader(args, train=False)
        prec1 = validate(val_loader, model, criterion)
        log.info('Test accuracy: {}'.format(prec1))

    cmanager.halt = True

    log.info('elapsed_time {0}'.format(elapsed_time))


def train(model, criterion, optimizer, batch_meter, data_meter, nn_meter,
          loader, epoch, itr, begin_time):

    losses = Meter(ptag='Loss')
    top1 = Meter(ptag='Prec@1')
    top5 = Meter(ptag='Prec@5')

    # switch to train mode
    model.train()

    # spoof sampler to continue from checkpoint w/o loading data all over again
    _train_loader = loader.__iter__()
    for i in range(itr):
        try:
            next(_train_loader.sample_iter)
        except Exception:
            # finished epoch but prempted before state was updated
            log.info('Loader spoof error attempt {}/{}'.format(i, len(loader)))
            return

    log.debug('Training (epoch {})'.format(epoch))

    if not args.bilat:
        model.communicator_warmup()
    if args.bilat:
        model.enable_gossip()

    batch_time = time.time()
    for i, (batch, target) in enumerate(_train_loader, start=itr):

        target = target.cuda(non_blocking=True)
        data_meter.update(time.time() - batch_time)

        # ----------------------------------------------------------- #
        # Forward/Backward pass
        # ----------------------------------------------------------- #
        nn_time = time.time()
        output = model(batch)
        loss = criterion(output, target)
        if args.bilat:
            bilat_freq = 100
            if i == 0:
                update_global_iteration_counter(itr=1,
                                                itr_per_epoch=len(loader))
                update_bilat_learning_rate(model, itr_per_epoch=len(loader))
            elif (i + args.rank) % (bilat_freq) == 0:
                update_global_iteration_counter(itr=bilat_freq,
                                                itr_per_epoch=len(loader))
                update_bilat_learning_rate(model, itr_per_epoch=len(loader))
        loss.backward()
        update_learning_rate(optimizer, epoch, itr=i,
                             itr_per_epoch=len(loader))
        optimizer.step()  # optimization update
        optimizer.zero_grad()
        if not args.overlap and not args.bilat:
            model.transfer_params()
        nn_meter.update(time.time() - nn_time)
        # ----------------------------------------------------------- #

        batch_meter.update(time.time() - batch_time)
        batch_time = time.time()

        log_time = time.time()
        # measure accuracy and record loss
        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), batch.size(0))
        top1.update(prec1.item(), batch.size(0))
        top5.update(prec5.item(), batch.size(0))
        if i % args.print_freq == 0:
            ep = epoch if not args.bilat else args.global_epoch
            itr = i if not args.bilat else args.global_itr % (len(loader) * args.world_size)
            with open(args.out_fname, '+a') as f:
                print('{ep},{itr},{bt},{nt},{dt},'
                      '{loss.val:.4f},{loss.avg:.4f},'
                      '{top1.val:.3f},{top1.avg:.3f},'
                      '{top5.val:.3f},{top5.avg:.3f},-1'
                      .format(ep=ep, itr=itr,
                              bt=batch_meter,
                              dt=data_meter, nt=nn_meter,
                              loss=losses, top1=top1,
                              top5=top5), file=f)
        log_time = time.time() - log_time
        log.debug(log_time)

    with open(args.out_fname, '+a') as f:
        print('{ep},{itr},{bt},{nt},{dt},'
              '{loss.val:.4f},{loss.avg:.4f},'
              '{top1.val:.3f},{top1.avg:.3f},'
              '{top5.val:.3f},{top5.avg:.3f},-1'
              .format(ep=epoch, itr=i,
                      bt=batch_meter,
                      dt=data_meter, nt=nn_meter,
                      loss=losses, top1=top1,
                      top5=top5), file=f)


def validate(val_loader, model, criterion):
    """ Evaluate model using criterion on validation set """

    losses = Meter(ptag='Loss')
    top1 = Meter(ptag='Prec@1')
    top5 = Meter(ptag='Prec@5')

    # switch to evaluate mode
    model.eval()

    if args.bilat:
        model.disable_gossip()

    with torch.no_grad():
        for i, (features, target) in enumerate(val_loader):

            target = target.cuda(non_blocking=True)

            # compute output
            output = model(features)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), features.size(0))
            top1.update(prec1.item(), features.size(0))
            top5.update(prec5.item(), features.size(0))

        log.info(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
                 .format(top1=top1, top5=top5))

    return top1.avg


def accuracy(output, target, topk=(1,)):
    """ Computes the precision@k for the specified values of k """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def update_state(state, update_dict):
    """ Helper function to update global state dict """
    for key in update_dict:
        state[key] = copy.deepcopy(update_dict[key])


def update_peers_per_itr(model, epoch):
    """ Update the model's peers per itr according to specified schedule """
    if args.single_threaded:
        return
    ppi = None
    e_max = -1
    for e in args.ppi_schedule:
        if e_max <= e and epoch >= e:
            e_max = e
            ppi = args.ppi_schedule[e]
    model.update_gossiper('peers_per_itr', ppi)


def update_bilat_learning_rate(model, itr_per_epoch=None, scale=1):
    """
    1) Linearly warmup to reference learning rate (5 epochs)
    2) Decay learning rate exponentially (epochs 30, 60, 80)
    ** note: args.lr is the reference learning rate from which to scale up
    ** note: minimum global batch-size is 256
    """
    if not args.bilat:
        return
    target_lr = args.lr * args.batch_size * scale * args.world_size / 256
    epoch = args.global_epoch
    itr_per_epoch *= args.world_size
    itr = args.global_itr % itr_per_epoch

    lr = None
    if args.warmup and epoch < 5:  # warmup to scaled lr
        if target_lr <= args.lr:
            lr = target_lr
        else:
            assert itr is not None and itr_per_epoch is not None
            count = epoch * itr_per_epoch + itr + 1
            incr = (target_lr - args.lr) * (count / (5 * itr_per_epoch))
            lr = args.lr + incr
    else:
        lr = target_lr
        for e in args.lr_schedule:
            if epoch >= e:
                lr *= args.lr_schedule[e]

    if lr is not None:
        model.update_lr(lr)


def update_global_iteration_counter(itr_per_epoch, itr=1):
    global args
    if not args.bilat:
        return

    p_str = ''
    for _ in range(itr):
        p_str += '-'
    # update global iteration counter
    with open(args.shared_fpath, '+a') as f:
        print(p_str, end='', file=f)

    # determine file length
    args.global_itr = int(os.stat(args.shared_fpath).st_size)
    args.global_epoch = int(args.global_itr / (itr_per_epoch * args.world_size))
    log.debug('global epoch estimate {}, global itr estimate {}'.format(
        args.global_epoch, args.global_itr))


def update_learning_rate(optimizer, epoch, itr=None, itr_per_epoch=None,
                         scale=1):
    """
    1) Linearly warmup to reference learning rate (5 epochs)
    2) Decay learning rate exponentially (epochs 30, 60, 80)
    ** note: args.lr is the reference learning rate from which to scale up
    ** note: minimum global batch-size is 256
    """
    target_lr = args.lr * args.batch_size * scale * args.world_size / 256
    if args.bilat:
        if args.global_itr is not None and args.global_epoch is not None:
            epoch = args.global_epoch
            itr_per_epoch *= args.world_size
            itr = args.global_itr % itr_per_epoch

    lr = None
    if args.warmup and epoch < 5:  # warmup to scaled lr
        if target_lr <= args.lr:
            lr = target_lr
        else:
            assert itr is not None and itr_per_epoch is not None
            count = epoch * itr_per_epoch + itr + 1
            incr = (target_lr - args.lr) * (count / (5 * itr_per_epoch))
            lr = args.lr + incr
    else:
        lr = target_lr
        for e in args.lr_schedule:
            if epoch >= e:
                lr *= args.lr_schedule[e]

    if lr is not None:
        log.debug('Updating learning rate to {}'.format(lr))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def make_dataloader(args, train=True):
    """ Returns train/val distributed dataloaders (cf. ImageNet in 1hr) """

    train_dir = TRAIN_DIR
    val_dir = VAL_DIR

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    if train:
        log.debug('fpaths train {}'.format(train_dir))
        train_dataset = datasets.ImageFolder(train_dir, transforms.Compose([
                            transforms.RandomResizedCrop(224),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            normalize]))

        # sampler produces indices used to assign data samples to each agent
        train_sampler = torch.utils.data.distributed.DistributedSampler(
                            dataset=train_dataset,
                            num_replicas=args.world_size,
                            rank=args.rank)

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size,
            shuffle=(train_sampler is None),
            num_workers=args.num_dataloader_workers,
            pin_memory=True, sampler=train_sampler)

        return train_loader, train_sampler

    else:
        log.debug('fpaths val {}'.format(val_dir))
        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(val_dir, transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize])),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_dataloader_workers, pin_memory=True)

        return val_loader


def parse_args():
    """
    Set env-vars and global args
        rank: <-- $SLRUM_PROCID
        world_size<-- $SLURM_NTASKS
        Master address <-- $SLRUM_NODENAME of rank 0 process (or HOSTNAME)
        Master port <-- any free port (doesn't really matter)
    """
    args = parser.parse_args()
    ClusterManager.set_user_name(args.user_name)

    args.distributed = True if args.distributed == 'True' else False
    if not args.distributed:
        args.rank = 0
        args.world_size = 1
        args.device_id = 0
        args.master_addr = 'localhost'
    else:
        args.master_addr = os.environ['HOSTNAME']
        if args.backend == 'mpi':
            args.rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
            args.world_size = int(os.environ['OMPI_UNIVERSE_SIZE'])
            args.device_id = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
        else:
            args.rank = int(os.environ['SLURM_PROCID'])
            args.world_size = int(os.environ['SLURM_NTASKS'])
            args.device_id = int(os.environ['SLURM_LOCALID'])
    args.out_fname = ClusterManager.CHECKPOINT_DIR \
        + args.tag \
        + 'out_r' + str(args.rank) \
        + '_n' + str(args.world_size) \
        + '.csv'
    args.resume = True if args.resume == 'True' else False
    args.verbose = True if args.verbose == 'True' else False
    args.train_fast = True if args.train_fast == 'True' else False
    args.nesterov = True if args.nesterov == 'True' else False
    args.checkpoint_all = True if args.checkpoint_all == 'True' else False
    args.warmup = True if args.warmup == 'True' else False
    args.cpu_comm = True if args.backend == 'tcp' else False
    args.comm_device = torch.device('cpu') if args.cpu_comm else torch.device('cuda')
    args.overlap = True if args.overlap == 'True' else False
    args.single_threaded = True if args.single_threaded == 'True' else False
    args.data_preloaded = True if args.data_preloaded == 'True' else False
    args.push_sum = True if args.push_sum == 'True' else False
    args.all_reduce = True if args.all_reduce == 'True' else False
    args.bilat = True if args.bilat == 'True' else False
    args.global_epoch = None
    args.global_itr = None
    args.lr_schedule = {}
    if args.schedule is None:
        args.schedule = [30, 0.1, 60, 0.1, 80, 0.1]
    i, epoch = 0, None
    for v in args.schedule:
        if i == 0:
            epoch = v
        elif i == 1:
            args.lr_schedule[epoch] = v
        i = (i + 1) % 2
    del args.schedule

    # parse peers per itr sched (epoch, num_peers)
    args.ppi_schedule = {}
    if args.peers_per_itr_schedule is None:
        args.peers_per_itr_schedule = [0, 1]
    i, epoch = 0, None
    for v in args.peers_per_itr_schedule:
        if i == 0:
            epoch = v
        elif i == 1:
            args.ppi_schedule[epoch] = v
        i = (i + 1) % 2
    del args.peers_per_itr_schedule
    # must specify how many peers to communicate from the start of training
    assert 0 in args.ppi_schedule
    if args.distributed:
        try:
            args.graph = GRAPH_TOPOLOGIES[args.graph_type](
                args.rank, args.world_size, peers_per_itr=args.ppi_schedule[0])
        except Exception:
            args.graph = None
        try:
            args.mixing = MIXING_STRATEGIES[args.mixing_strategy](args.graph)
        except Exception:
            args.mixing = None
    else:
        args.graph, args.mixing = None, None

    return args


def init_model():
    """
    Initialize resnet50 similarly to "ImageNet in 1hr" paper
        Batch norm moving average "momentum" <-- 0.9
        Fully connected layer <-- Gaussian weights (mean=0, std=0.01)
        gamma of last Batch norm layer of each residual block <-- 0
    """
    model = models.resnet50()
    for m in model.modules():
        if isinstance(m, Bottleneck):
            num_features = m.bn3.num_features
            m.bn3.weight = Parameter(torch.zeros(num_features))
    model.fc.weight.data.normal_(0, 0.01)
    return model


if __name__ == '__main__':
    mp.set_start_method('forkserver', force=True)
    main()
    print('hello world')
