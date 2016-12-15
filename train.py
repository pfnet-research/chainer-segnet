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
from chainer.dataset import convert
from chainer.training import updater as default_updater
from chainer.training import extensions
from lib import updater as updater_module
from lib import CamVid
from lib import create_logger
from lib import create_result_dir
from lib import get_args
from lib import get_model
from lib import get_optimizer

import json
import logging
import os


class TestModeEvaluator(extensions.Evaluator):

    def __init__(self, iterator, target, converter=convert.concat_examples,
                 device=None, eval_hook=None, eval_func=None):
        super(TestModeEvaluator, self).__init__(
            iterator, target, converter, eval_hook, eval_func)

    def evaluate(self):
        model = self.get_target('main')
        model.train = False
        ret = super(TestModeEvaluator, self).evaluate()
        model.train = True
        return ret

if __name__ == '__main__':
    args = get_args()
    if args.result_dir is None:
        result_dir = create_result_dir(args.model_name)
    else:
        if not os.path.exists(args.result_dir):
            os.makedirs(args.result_dir)
        result_dir = args.result_dir
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
        optimizer, args.n_encdec, args.n_mid, args.finetune, True, result_dir)
    if args.resume is not None:
        serializers.load_npz(args.resume, model)

    # Prepare devices
    devices = {}
    for gid in [int(i) for i in args.gpus.split(',')]:
        if 'main' not in devices:
            devices['main'] = gid
        else:
            devices['gpu{}'.format(gid)] = gid
    if devices['main'] >= 0:
        model.to_gpu(devices['main'])

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

    if not args.finetune:
        updater = updater_module.Updater(
            model, train_iter, devices, args.n_encdec)
        updater.depth = args.train_depth
    else:
        logging.info('model links:')
        for name, _ in model.namedlinks():
            logging.info(name)
        optimizer.setup(model)
        updater = default_updater.ParallelUpdater(
            train_iter, optimizer, devices=devices)

    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=result_dir)

    # Add Evaluator
    trainer.extend(
        TestModeEvaluator(valid_iter, model, device=devices['main']),
        trigger=(args.valid_freq, 'epoch'))

    # Add dump_graph
    graph_name = 'encdec{}.dot'.format(args.train_depth) \
        if not args.finetune else 'encdec_finetune.dot'
    trainer.extend(extensions.dump_graph('main/loss', out_name=graph_name))

    # Add snapshot_object
    if not args.finetune:
        model_save_fn = 'encdec{.updater.depth}'.format(trainer) + \
            '_epoch_{.updater.epoch}.model'
    else:
        model_save_fn = 'encdec4_finetune_epoch_' + '{.updater.epoch}.model'
    trainer.extend(
        extensions.snapshot_object(
            model, trigger=(args.snapshot_epoch, 'epoch'),
            filename=model_save_fn))

    # Add Logger
    if not args.finetune:
        log_fn = 'log_encdec{}'.format(args.train_depth)
    else:
        log_fn = 'log_encdec_finetune'
    trainer.extend(
        extensions.LogReport(
            trigger=(args.show_log_iter, 'iteration'),
            log_name=log_fn))
    trainer.extend(extensions.PrintReport(
        ['epoch', 'iteration', 'main/loss', 'validation/main/loss']))
    trainer.extend(extensions.ProgressBar())

    trainer.run()
