# Copyright (c) 2016 Shunta Saito

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from chainer import optimizers

import chainer
import imp
import logging
import os
import shutil
import sys
import time


def create_result_dir(model_name):
    result_dir = 'results/{}_{}'.format(
        model_name, time.strftime('%Y-%m-%d_%H-%M-%S'))
    if os.path.exists(result_dir):
        result_dir += '_{}'.format(time.clock())
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    return result_dir


def create_logger(args, result_dir):
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    msg_format = '%(asctime)s [%(levelname)s] %(message)s'
    formatter = logging.Formatter(msg_format)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    root.addHandler(ch)
    fileHandler = logging.FileHandler("{}/stdout.log".format(result_dir))
    fileHandler.setFormatter(formatter)
    root.addHandler(fileHandler)
    logging.info(sys.version_info)
    logging.info('chainer version: {}'.format(chainer.__version__))
    logging.info('cuda: {}, cudnn: {}'.format(
        chainer.cuda.available, chainer.cuda.cudnn_enabled))
    logging.info(args)


def get_model(
        model_file, model_name, loss_file, loss_name, class_weight, n_encdec,
        n_classes, in_channel, n_mid, train_depth=None, result_dir=None):
    model = imp.load_source(model_name, model_file)
    model = getattr(model, model_name)
    loss = imp.load_source(loss_name, loss_file)
    loss = getattr(loss, loss_name)

    # Initialize
    model = model(n_encdec, n_classes, in_channel, n_mid)
    if train_depth:
        model = loss(model, class_weight, train_depth)

    # Copy files
    if result_dir is not None:
        base_fn = os.path.basename(model_file)
        dst = '{}/{}'.format(result_dir, base_fn)
        if not os.path.exists(dst):
            shutil.copy(model_file, dst)
        base_fn = os.path.basename(loss_file)
        dst = '{}/{}'.format(result_dir, base_fn)
        if not os.path.exists(dst):
            shutil.copy(loss_file, dst)

    return model


def get_optimizer(opt, lr=None, adam_alpha=None, adam_beta1=None,
                  adam_beta2=None, adam_eps=None, weight_decay=None):
    if opt == 'MomentumSGD':
        optimizer = optimizers.MomentumSGD(lr=lr, momentum=0.9)
    elif opt == 'Adam':
        optimizer = optimizers.Adam(
            alpha=adam_alpha, beta1=adam_beta1,
            beta2=adam_beta2, eps=adam_eps)
    elif opt == 'AdaGrad':
        optimizer = optimizers.AdaGrad(lr=lr)
    elif opt == 'RMSprop':
        optimizer = optimizers.RMSprop(lr=lr)
    else:
        raise Exception('No optimizer is selected')

    # The first model as the master model
    if opt == 'MomentumSGD':
        optimizer.decay = weight_decay

    return optimizer
