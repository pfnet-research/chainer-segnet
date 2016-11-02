#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2016 Shunta Saito

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import chainer
import numpy


def get_args():
    parser = argparse.ArgumentParser()

    # Training settings
    parser.add_argument(
        '--model_file', type=str, default='models/segnet.py',
        help='The model filename with .py extension')
    parser.add_argument(
        '--model_name', type=str, default='SegNet',
        help='The model class name')
    parser.add_argument(
        '--loss_file', type=str, default='models/segnet.py',
        help='The filename that contains the definition of loss function')
    parser.add_argument(
        '--loss_name', type=str, default='SegNetLoss',
        help='The name of the loss function')
    parser.add_argument(
        '--resume', type=str,
        help='Saved trainer state to be used for resuming')
    parser.add_argument(
        '--trunk', type=str,
        help='The pre-trained trunk param filename with .npz extension')
    parser.add_argument(
        '--epoch', type=int, default=1000,
        help='When the trianing will finish')
    parser.add_argument(
        '--gpus', type=str, default='0',
        help='GPU Ids to be used')
    parser.add_argument(
        '--batchsize', type=int, default=64,
        help='minibatch size')
    parser.add_argument(
        '--snapshot_iter', type=int, default=10000,
        help='The current learnt parameters in the model is saved every'
             'this iteration')
    parser.add_argument(
        '--valid_freq', type=int, default=1,
        help='Perform test every this iteration (0 means no test)')
    parser.add_argument(
        '--valid_batchsize', type=int, default=16,
        help='The mini-batch size during validation loop')
    parser.add_argument(
        '--show_log_iter', type=int, default=10,
        help='Show loss value per this iterations')

    # Settings
    parser.add_argument(
        '--shift_jitter', type=int, default=50,
        help='Shift jitter amount for data augmentation (typically 50)')
    parser.add_argument(
        '--scale_jitter', type=float, default=0.2,
        help='Scale jitter amount for data augmentation '
             '(typically 0.2 for 0.8~1.2)')
    parser.add_argument(
        '--fliplr', action='store_true', default=False,
        help='Perform LR flipping during training for data augmentation')
    parser.add_argument(
        '--rotate', action='store_true', default=False,
        help='Perform rotation for data argumentation')
    parser.add_argument(
        '--rotate_max', type=float, default=7,
        help='Maximum roatation angle(degree) for data augmentation')
    parser.add_argument(
        '--scale', type=float, default=1.0,
        help='Scale for the input images and label images')
    parser.add_argument(
        '--n_classes', type=int, default=12,
        help='The number of classes that the model predicts')
    parser.add_argument(
        '--mean', type=str, default=None,
        help='Mean npy over the training data')
    parser.add_argument(
        '--std', type=str, default=None,
        help='Stddev npy over the training data')
    parser.add_argument(
        '--class_weights', type=str,
        default='0.2595,0.1826,4.5640,0.1417,0.9051,0.3826,9.6446,1.8418,'
                '0.6823,6.2478,7.3614',
        help='Weights for classes used in softmax cross entropy loss '
             'calculation')

    # Dataset paths
    parser.add_argument(
        '--train_img_dir', type=str, default='data/train',
        help='Full path to images for trianing')
    parser.add_argument(
        '--valid_img_dir', type=str, default='data/val',
        help='Full path to images for validation')
    parser.add_argument(
        '--train_lbl_dir', type=str, default='data/trainannot',
        help='Full path to labels for trianing')
    parser.add_argument(
        '--valid_lbl_dir', type=str, default='data/valannot',
        help='Full path to labels for validation')
    parser.add_argument(
        '--train_list_fn', type=str, default='data/train.txt',
        help='Full path to the file list of training images')
    parser.add_argument(
        '--valid_list_fn', type=str, default='data/val.txt',
        help='Full path to the file list of validation images')

    # Optimization settings
    parser.add_argument(
        '--opt', type=str, default='Adam',
        choices=['MomentumSGD', 'Adam', 'AdaGrad', 'RMSprop'],
        help='Optimization method')
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--adam_alpha', type=float, default=0.00001)
    parser.add_argument('--adam_beta1', type=float, default=0.9)
    parser.add_argument('--adam_beta2', type=float, default=0.999)
    parser.add_argument('--adam_eps', type=float, default=1e-8)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument(
        '--lr_decay_freq', type=int, default=10,
        help='The learning rate will be decreased every this epoch')
    parser.add_argument(
        '--lr_decay_ratio', type=float, default=0.1,
        help='When the learning rate is decreased, this number will be'
             'multiplied')
    parser.add_argument('--seed', type=int, default=1701)
    args = parser.parse_args()
    args.class_weights = [float(v) for v in args.class_weights.split(',')
                          if len(v) > 0]
    xp = chainer.cuda.cupy if chainer.cuda.available else numpy
    xp.random.seed(args.seed)
    numpy.random.seed(args.seed)
    return args
