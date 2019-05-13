# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
Visualization script
"""

import re
import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pylab as plt
from matplotlib import cm


font_size = 20
font = {'size': font_size}
matplotlib.rc('font', **font)


def get_transformer_config():
    nodes = [8, 8]
    fpaths, tags, legends, colors = [], [], [], []
    fpath = 'results_dir/transformer_{tag}_test.out'
    ps_tags = [
        'ps_sm',
        'ps'
    ]
    ps_legends = ['SGP (25K batch)', 'SGP (400K batch)']
    ps_colors = [cm.Blues(x) for x in np.linspace(0.3, 0.8, len(ps_tags))]
    fpaths.append(fpath)
    tags.append(ps_tags)
    legends.append(ps_legends)
    colors.append(ps_colors)
    fpath = 'results_dir/transformer_{tag}_test.out'
    ar_tags = [
        'ar_sm',
        'ar'
    ]
    ar_legends = ['SGD (25K batch)', 'SGD (400K batch)']
    ar_colors = [cm.Reds(x) for x in np.linspace(0.3, 0.8, len(ar_tags))]
    fpaths.append(fpath)
    tags.append(ar_tags)
    legends.append(ar_legends)
    colors.append(ar_colors)

    return nodes, fpaths, tags, legends, colors


def get_ib_config():
    nodes = [4, 8, 16, 32]
    fpaths, tags, legends, colors = [], [], [], []
    ps_ib_fpath = 'results_dir/out_files/{tag}out_r{r}_n{n}.csv'
    ps_ib_tags = [
        'SIMPLE-PS-SGD-4IB',
        'SIMPLE-PS-SGD-8IB',
        'SIMPLE-PS-SGD-16IB',
        'SIMPLE-PS-SGD-32IB-4'
    ]
    ps_ib_legends = ['SGP 4 nodes', 'SGP 8 nodes', 'SGP 16 nodes', 'SGP 32 nodes']
    ps_ib_colors = [cm.Blues(x) for x in np.linspace(0.3, 0.8, len(ps_ib_tags))]
    fpaths.append(ps_ib_fpath)
    tags.append(ps_ib_tags)
    legends.append(ps_ib_legends)
    colors.append(ps_ib_colors)
    ar_ib_fpath = 'results_dir/out_files/{tag}out_r{r}_n{n}.csv'
    ar_ib_tags = [
        'AR-DPSGD-CPUCOMM-4IB-SCRATCh',
        'AR-DPSGD-CPUCOMM-8IB-SCRATCh',
        'AR-DPSGD-CPUCOMM-16IB-SCRATCh',
        'AR-DPSGD-CPUCOMM-32IB-SCRATCH'
    ]
    ar_ib_legends = ['AR-SGD 4 nodes', 'AR-SGD 8 nodes', 'AR-SGD 16 nodes', 'AR-SGD 32 nodes']
    ar_ib_colors = [cm.Reds(x) for x in np.linspace(0.3, 0.8, len(ar_ib_tags))]
    fpaths.append(ar_ib_fpath)
    tags.append(ar_ib_tags)
    legends.append(ar_ib_legends)
    colors.append(ar_ib_colors)

    return nodes, fpaths, tags, legends, colors


def get_eth_config():
    nodes = [4, 8, 16, 32]
    fpaths, tags, legends, colors = [], [], [], []
    ar_eth_fpath = 'results_dir/out_files/{tag}out_r{r}_n{n}.csv'
    ar_eth_tags = [
        'AR-DPSGD-CPUCOMM-4ETH-SCRATCH',
        'AR-DPSGD-CPUCOMM-8ETH-SCRATCH',
        'AR-DPSGD-CPUCOMM-16ETH-SCRATCH',
        'AR-DPSGD-CPUCOMM-32ETH-SCRATCH'
    ]
    ar_eth_legends = ['AR-SGD 4 nodes', 'AR-SGD 8 nodes', 'AR-SGD 16 nodes', 'AR-SGD 32 nodes']
    ar_eth_colors = [cm.Reds(x) for x in np.linspace(0.3, 0.8, len(ar_eth_tags))]
    fpaths.append(ar_eth_fpath)
    tags.append(ar_eth_tags)
    legends.append(ar_eth_legends)
    colors.append(ar_eth_colors)

    ds_eth_fpath = 'results_dir/out_files/{tag}out_r{r}_n{n}.csv'
    ds_eth_tags = [
        'SDS-SGD-4ETH',
        'SDS-SGD-8ETH',
        'SDS-SGD-16ETH',
        'SDS-SGD-32ETH'
    ]
    ds_eth_legends = ['D-PSGD 4 nodes', 'D-PSGD 8 nodes', 'D-PSGD 16 nodes', 'D-PSGD 32 nodes']
    ds_eth_colors = [cm.Greens(x) for x in np.linspace(0.3, 0.8, len(ds_eth_tags))]
    fpaths.append(ds_eth_fpath)
    tags.append(ds_eth_tags)
    legends.append(ds_eth_legends)
    colors.append(ds_eth_colors)

    ps_eth_fpath = 'results_dir/out_files/{tag}out_r{r}_n{n}.csv'
    ps_eth_tags = [
        'PS-SGD-4ETH-CHORD',
        'PS-SGD-8ETH-CHORD',
        'PS-SGD-16ETH-CHORD',
        'PS-SGD-32ETH-SCRATCH-CHORD'
    ]
    ps_eth_legends = ['SGP 4 nodes', 'SGP 8 nodes', 'SGP 16 nodes', 'SGP 32 nodes']
    ps_eth_colors = [cm.Blues(x) for x in np.linspace(0.3, 0.8, len(ps_eth_tags))]
    fpaths.append(ps_eth_fpath)
    tags.append(ps_eth_tags)
    legends.append(ps_eth_legends)
    colors.append(ps_eth_colors)

    # return nodes, fpaths[::-1], tags[::-1], legends[::-1], colors[::-1]
    return nodes, fpaths, tags, legends, colors


def parse_transformer_out(world_size, tag, fpath, itr_scale=1):
    f_fpath = fpath.format(tag=tag)
    itr_list, ppl_list, nll_list = [[] for _ in range(world_size)], [[] for _ in range(world_size)], [[] for _ in range(world_size)]
    time_list = [[0. for _ in range(100)] for _ in range(world_size)]
    with open(f_fpath, 'r') as f:
        for line in f:
            s1_line = line + ''
            s2_line = line + ''
            if re.search('train_wall', s1_line):
                line = s1_line.split('|')
                rank = int(line[0].split(' ')[0].replace(':', ''))
                try:
                    ep = int(line[1].split(' ')[-2])
                except Exception:
                    continue
                if ep == 1: continue  # skip first epoch
                time = float(line[-1].split(' ')[-1].replace('\n', ''))
                if time > time_list[rank][ep - 2]:
                    time_list[rank][ep - 2] = time
            elif re.search('valid_nll_loss', s2_line):
                line = s2_line.split('|')
                rank = int(line[0].split(' ')[0].replace(':', ''))
                ep = int(line[1].split(' ')[-2])
                if ep == 1: continue  # skip first epoch
                itr = int(line[-2].split(' ')[-2])
                ppl = float(line[-3].split(' ')[-2])
                nll = float(line[-4].split(' ')[-2])
                itr_list[rank].append(itr)
                ppl_list[rank].append(ppl)
                nll_list[rank].append(nll)

    pdf = pd.DataFrame()
    itr_columns, ppl_columns, nll_columns, time_columns = [], [], [], []
    itr_len = min([len(itr_lr) for itr_lr in itr_list if len(itr_lr) != 0])
    for r in range(world_size):
        if len(itr_list[r]) == 0:
            continue
        rtag = 'itr' + str(r)
        pdf[rtag] = itr_list[r][:itr_len]
        itr_columns.append(rtag)
        rtag = 'ppl' + str(r)
        pdf[rtag] = ppl_list[r][:itr_len]
        ppl_columns.append(rtag)
        rtag = 'nll' + str(r)
        pdf[rtag] = nll_list[r][:itr_len]
        nll_columns.append(rtag)
        rtag = 'time' + str(r)
        pdf[rtag] = time_list[r][:itr_len]
        time_columns.append(rtag)

    pdf['itr'] = pdf[itr_columns].mean(axis=1)
    pdf['ppl'] = pdf[ppl_columns].mean(axis=1)
    pdf['nll'] = pdf[nll_columns].mean(axis=1)
    pdf['time'] = pdf[time_columns].mean(axis=1)
    print(pdf.head())
    return pdf


def parse_csv(world_size, tag, fpath):
    itrs = {4: 1251, 8: 625, 16: 312, 32: 156}
    itr = itrs[world_size]

    train_rtags, val_rtags, time_rtags = [], [], []
    pdf = pd.DataFrame()
    for r in range(world_size):
        # - load csv
        csv_fpath = fpath.format(tag=tag, r=r, n=world_size)
        df = pd.read_csv(csv_fpath, skiprows=4).drop_duplicates()
        # - update global df with node train statistic
        rtag = 'train:' + str(r)
        pdf[rtag] = 100 - df[df['itr'] == itr]['avg:Prec@1']
        train_rtags.append(rtag)
        # - update global df with node val statistic
        try:
            rtag = 'val:' + str(r)
            pdf[rtag] = 100 - df[df['val'] != -1]['val'].values
            val_rtags.append(rtag)
        except:
            pass
        # - update global df with timing statistic
        rtag = 'time:' + str(r)
        pdf[rtag] = df[df['itr'] == itr]['avg:BT(s)']
        time_rtags.append(rtag)

    # - compute mean statistic across all nodes
    pdf['train_mean'] = pdf[train_rtags].mean(axis=1)
    pdf['val_mean'] = pdf[val_rtags].mean(axis=1)
    pdf['time_mean'] = pdf[time_rtags].mean(axis=1)
    pdf['itr'] = [itr * ep for ep in range(1, pdf.shape[0] + 1)]
    pdf['time'] = pdf['itr'].values * pdf['time_mean'].iloc[-1]
    print(pdf.head())
    return pdf


def plot_transformer(save_fname='transformer.pdf'):
    nodes, fpaths, tags, legends, colors = get_transformer_config()

    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    for n, ntags, nlegends, ncolors in zip(nodes, zip(*tags), zip(*legends),
                                           zip(*colors)):  # iterate over nodes
        for i in range(len(ntags)):  # iterate over algs
            df = parse_transformer_out(n, ntags[i], fpaths[i])
            label = nlegends[i]
            df.plot(x='itr', y='nll', ax=ax, color=ncolors[i],
                    grid=True, label=label, fontsize=14,
                    xlim=(1000, 25000), ylim=(2.0, 3.0))

    ax.set_ylabel('Validation Loss (NLL)', fontsize=font_size)
    ax.set_xlabel('Opt. steps', fontsize=font_size)
    ax.grid(which='both')
    ax.grid(which='minor', alpha=0.2)
    ax.grid(which='major', alpha=0.5)
    plt.tight_layout()
    fig.savefig(save_fname)
    plt.show()


def plot_itrs(save_fname='itr.pdf', val=False):
    nodes, fpaths, tags, legends, colors = get_eth_config()

    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    for n, ntags, nlegends, ncolors in zip(nodes, zip(*tags), zip(*legends),
                                           zip(*colors)):  # iterate over nodes
        if n != 16 and n != 8:
            continue
        for i in range(len(ntags)):  # iterate over algs
            df = parse_csv(n, ntags[i], fpaths[i])
            color = ncolors[i]
            label = nlegends[i]
            if n == 8:
                linestyle = '--'
            else:
                linestyle = '-'
            if val:
                df.plot(x='time', y='val_mean', ax=ax, color=color,
                        grid=True, label=label, fontsize=16)
            else:
                df.plot(x='time', y='train_mean', ax=ax, color=color,
                        grid=True, label=label, fontsize=16,
                        linestyle=linestyle)

    if val:
        ax.set_ylabel('Validation Error (%)', fontsize=font_size)
    else:
        ax.set_ylabel('Training Error (%)', fontsize=font_size)
    ax.set_xlabel('Time (s)', fontsize=font_size)
    # ax.set_xlabel('Iterations', fontsize=font_size)
    ax.grid(which='both')
    ax.grid(which='minor', alpha=0.2)
    ax.grid(which='major', alpha=0.5)
    plt.legend(prop={'size': 16})
    plt.tight_layout()
    fig.savefig(save_fname)
    plt.show()


def plot_scaling(save_fname='scaling.pdf'):
    bspn = 256  # batch-size per node
    pdf = pd.DataFrame()

    net_tags = ['eth', 'ib']
    # net_tags = ['ib', 'eth']
    net_columns = {}
    # construct scaling data-frame
    for net_tag in net_tags:
        if net_tag == 'eth':
            nodes, fpaths, tags, legends, colors = get_eth_config()
        elif net_tag == 'ib':
            nodes, fpaths, tags, legends, colors = get_ib_config()
        pdf['nodes'] = nodes
        columns = []
        for alg_tags, alg_fpath, alg_legends in zip(tags, fpaths, legends):  # iterate over algorithms
            alg = alg_legends[0].split(' ')[0]  # algorithm name
            alg_thpt = []
            for i, n in enumerate(nodes):
                df = parse_csv(n, alg_tags[i], alg_fpath)
                tpi = df['time_mean'].dropna().iloc[-1]  # avg time per itr
                thpt = tpi
                # thpt = (bspn * n) / tpi  # throughput (images per second)
                alg_thpt.append(thpt)
            net_label = '(InfiniBand)' if net_tag == 'ib' else '(Ethernet)'
            rtag = alg + ' ' + net_label
            pdf[rtag] = alg_thpt
            columns.append(rtag)
        net_columns[net_tag] = columns

    # plot scaling data
    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    for net_tag in net_tags:
        if net_tag == 'eth':
            _, _, _, _, colors = get_eth_config()
        elif net_tag == 'ib':
            _, _, _, _, colors = get_ib_config()
        columns = net_columns[net_tag]
        print(columns)
        linestyle = '--' if net_tag == 'ib' else '-'
        marker = '1' if net_tag == 'ib' else '2'
        for alg, alg_colors in zip(columns, colors):
            color = alg_colors[-2]
            pdf.plot(x='nodes', y=alg, ax=ax, color=color, grid=True,
                     label=alg, marker=marker, xticks=nodes, ms=15,
                     subplots=True, linestyle=linestyle,
                     fontsize=16)
    ax.set_xlabel('Number of nodes', fontsize=font_size)
    ax.set_ylabel('Time per iteration (s)', fontsize=font_size)
    # ax.set_ylabel('Throughput (Images/sec.)', fontsize=font_size)
    ax.grid(which='both')
    ax.grid(which='minor', alpha=0.2)
    ax.grid(which='major', alpha=0.5)
    plt.legend(prop={'size': 16})
    plt.tight_layout()
    fig.savefig(save_fname)
    plt.show()


def main():
    # plot_itrs(val=False)
    plot_scaling()
    # plot_transformer()


if __name__ == "__main__":
    main()
