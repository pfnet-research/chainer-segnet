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
from chainer.training import updater
from lib import CamVid
from lib import create_logger
from lib import create_result_dir
from lib import get_args
from lib import get_model
from lib import get_optimizer

import json
import logging
import os


@training.make_extension(default_name='recover_links')
def recover_links(trainer):
    model = trainer.updater.get_optimizer('main').target
    n_encdec = model.predictor.n_encdec
    train_depth = model.train_depth
    for d in range(1, n_encdec + 1):
        if d != train_depth:
            model.predictor.recover_link('encdec{}'.format(d))
    model.predictor.recover_link('conv_cls')


@training.make_extension(default_name='remove_links')
def remove_links(trainer):
    model = trainer.updater.get_optimizer('main').target
    n_encdec = model.predictor.n_encdec
    train_depth = model.train_depth
    for d in range(1, n_encdec + 1):
        if d != train_depth:
            if model.predictor.is_registered_link('encdec{}'.format(d)):
                model.predictor.remove_link('encdec{}'.format(d))
    if train_depth > 1 and model.predictor.is_registered_link('conv_cls'):
        model.predictor.remove_link('conv_cls')

if __name__ == '__main__':
    args = get_args()
    if args.result_dir is None:
        result_dir = create_result_dir(args.model_name)
    else:
        if not os.path.exists(args.result_dir):
            os.makedirs(args.result_dir)
        result_dir = args.result_dir
    json.dump(vars(args), open('{}/args_{}.json'.format(
        result_dir, args.train_depth), 'w'))
    create_logger(args, result_dir)

    # Instantiate model
    model = get_model(
        args.model_file, args.model_name, args.loss_file, args.loss_name,
        args.class_weight if args.use_class_weight else None, args.n_encdec,
        args.n_classes, args.in_channel, args.n_mid, args.train_depth,
        result_dir)

    # Initialize optimizer
    optimizer = get_optimizer(
        args.opt, args.lr, adam_alpha=args.adam_alpha,
        adam_beta1=args.adam_beta1, adam_beta2=args.adam_beta2,
        adam_eps=args.adam_eps, weight_decay=args.weight_decay)
    optimizer.setup(model)

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
    logging.info('train: {}, valid: {}'.format(len(train), len(valid)))

    # Create iterators
    train_iter = iterators.MultiprocessIterator(
        train, args.batchsize, n_prefetch=10)
    valid_iter = iterators.SerialIterator(valid, args.valid_batchsize,
                                          repeat=False, shuffle=False)

    # Create updater
    updater = updater.ParallelUpdater(train_iter, optimizer, devices=devices)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=result_dir)
    if args.resume is not None:
        serializers.load_npz(args.resume, trainer)

    # Add Evaluator
    trainer.extend(
        extensions.Evaluator(valid_iter, model, device=devices['main']),
        trigger=(args.valid_freq, 'epoch'))

    # Add dump_graph
    graph_name = 'encdec{}.dot'.format(args.train_depth) \
        if not args.finetune else 'encdec_finetune.dot'
    trainer.extend(extensions.dump_graph('main/loss', out_name=graph_name))

    # Add snapshot_object
    if not args.finetune:
        save_fn = 'encdec{}'.format(args.train_depth) + \
            '_epoch_{.updater.epoch}.trainer'
    else:
        save_fn = 'encdec4_finetune_epoch_' + '{.updater.epoch}.trainer'
    trainer.extend(extensions.snapshot(
        filename=save_fn, trigger=(args.snapshot_epoch, 'epoch')),
        priority=0, invoke_before_training=False)

    # Add Logger
    if not args.finetune:
        log_fn = 'log_encdec{}.0'.format(args.train_depth)
    else:
        log_fn = 'log_encdec_finetune.0'
    if os.path.exists('{}/{}'.format(result_dir, log_fn)):
        n = int(log_fn.split('.')[-1])
        log_fn = log_fn.replace(str(n), str(n + 1))
    trainer.extend(extensions.ProgressBar())
    if args.show_log_iter:
        log_trigger = args.show_log_iter, 'iteration'
    else:
        log_trigger = 1, 'epoch'
    trainer.extend(extensions.LogReport(trigger=log_trigger, log_name=log_fn))
    trainer.extend(extensions.PrintReport(
        ['epoch', 'iteration', 'main/loss', 'validation/main/loss']))

    # Add remover and recoverer
    if not args.finetune:
        trainer.extend(remove_links, trigger=(args.snapshot_epoch, 'epoch'),
                       priority=500, invoke_before_training=True)
        trainer.extend(recover_links, trigger=(args.snapshot_epoch, 'epoch'),
                       priority=400, invoke_before_training=False)

    trainer.run()
