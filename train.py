#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2016 Shunta Saito

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from chainer import iterators
from chainer import serializers
from chainer import training
from chainer.training import extensions
from lib import updater as updater_module
from lib import CamVid
from lib import create_logger
from lib import create_result_dir
from lib import get_args
from lib import get_model
from lib import get_optimizer

import json
import six

if __name__ == '__main__':
    args = get_args()
    result_dir = create_result_dir(args.model_name)
    json.dump(vars(args), open('{}/args.json'.format(result_dir), 'w'))
    create_logger(args, result_dir)

    # Initialize optimizer
    optimizer = get_optimizer(
        args.opt, args.lr, adam_alpha=args.adam_alpha,
        adam_beta1=args.adam_beta1, adam_beta2=args.adam_beta2,
        adam_eps=args.adam_eps, weight_decay=args.weight_decay)

    # Instantiate model
    model = get_model(
        args.model_file, args.model_name, args.loss_file, args.loss_name,
        args.n_classes, args.class_weights if args.use_class_weights else None,
        optimizer, args.n_encdec, True, result_dir)

    # Prepare devices
    devices = {}
    for gid in [int(i) for i in args.gpus.split(',')]:
        if 'main' not in devices:
            devices['main'] = gid
        else:
            devices['gpu{}'.format(gid)] = gid

    # Setting up datasets
    train = CamVid(
        args.train_img_dir, args.train_lbl_dir, args.train_list_fn, args.mean,
        args.std, args.shift_jitter, args.scale_jitter, args.fliplr,
        args.rotate, args.rotate_max, args.scale, args.ignore_labels)
    valid = CamVid(
        args.valid_img_dir, args.valid_lbl_dir, args.valid_list_fn, args.mean,
        args.std, ignore_labels=args.ignore_labels)
    print('train: {}, valid: {}'.format(len(train), len(valid)))

    train_iter = iterators.MultiprocessIterator(train, args.batchsize)
    valid_iter = iterators.SerialIterator(valid, args.valid_batchsize,
                                          repeat=False, shuffle=False)

    updater = updater_module.Updater(
        model, train_iter, devices, args.n_encdec)
    updater.depth = args.train_depth

    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=result_dir)
    if args.resume is not None:
        serializers.load_npz(args.resume, trainer)

    # trainer.extend(
    #     extensions.Evaluator(valid_iter, model, device=devices['main']),
    #     trigger=(args.valid_freq, 'epoch'))
    trainer.extend(extensions.dump_graph('main/loss'))
    trainer.extend(
        extensions.snapshot(
            trigger=(args.snapshot_epoch, 'epoch'),
            filename='EncDec{.updater.depth}'.format(trainer) +
                     '_epoch_{.updater.epoch}'))
    trainer.extend(
        extensions.LogReport(trigger=(args.show_log_iter, 'iteration')))
    trainer.extend(extensions.PrintReport(
        ['epoch', 'iteration', 'main/loss', 'validation/main/loss']))
    trainer.extend(extensions.ProgressBar())

    trainer.run()
