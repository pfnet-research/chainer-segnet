#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import defaultdict

import glob
import json
import numpy as np
import os
import re

if True:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt


def show_loss_curve(log_fn):
    train_loss = defaultdict(list)
    valid_loss = []
    for l in json.load(open(log_fn)):
        train_loss[l['epoch']].append(l['main/loss'])
        if 'validation/main/loss' in l:
            valid_loss.append((l['epoch'], l['validation/main/loss']))
    train_loss = np.asarray([[epoch, np.mean(train_loss[epoch])]
                             for epoch in sorted(train_loss.keys())])
    if len(valid_loss) > 0:
        valid_loss = np.asarray(valid_loss)

    d = re.search('encdec([0-9]+)', log_fn)
    if d:
        d = '_' + d.groups()[0]
    else:
        d = '_4'
    args = json.load(open('{}/args{}.json'.format(os.path.dirname(log_fn), d)))
    axes[0].plot(
        train_loss[:, 0], train_loss[:, 1],
        label='Train {}({}) CW:{}'.format(
            args['opt'], args['adam_alpha']
            if args['opt'] == 'Adam' else args['lr'],
            args['use_class_weight']))

    if len(valid_loss) > 0:
        axes[1].plot(
            valid_loss[:, 0], valid_loss[:, 1],
            label='Valid {}({}) CW:{}'.format(
                args['opt'], args['adam_alpha']
                if args['opt'] == 'Adam' else args['lr'],
                args['use_class_weight']))

if __name__ == '__main__':
    fig, axes = plt.subplots(2)
    fig.set_size_inches(8, 10)
    for ax in axes:
        ax.set_ylabel('loss')
        ax.set_xlabel('epoch')

    for rdir in glob.glob('results*/log_encdec*'):
        show_loss_curve(rdir)
    axes[0].legend(loc='upper left', bbox_to_anchor=(1, 1))
    axes[1].legend(loc='upper left', bbox_to_anchor=(1, 1))

    plt.savefig('loss.png')
