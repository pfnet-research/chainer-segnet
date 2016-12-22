# Copyright (c) 2016 Shunta Saito

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import chainer
import numpy
import random


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
        '--result_dir', type=str, default=None,
        help='The directory path to store files')
    parser.add_argument(
        '--loss_file', type=str, default='models/segnet.py',
        help='The filename that contains the definition of loss function')
    parser.add_argument(
        '--loss_name', type=str, default='SegNetLoss',
        help='The name of the loss function')
    parser.add_argument(
        '--resume', type=str, default=None,
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
        '--batchsize', type=int, default=8,
        help='minibatch size')
    parser.add_argument(
        '--snapshot_epoch', type=int, default=10,
        help='The current learnt parameters in the model is saved every'
             'this epoch')
    parser.add_argument(
        '--valid_freq', type=int, default=10,
        help='Perform test every this epoch (0 means no test)')
    parser.add_argument(
        '--valid_batchsize', type=int, default=16,
        help='The mini-batch size during validation loop')
    parser.add_argument(
        '--show_log_iter', type=int, default=None,
        help='Show loss value per this iterations')

    # Settings
    parser.add_argument(
        '--train_depth', type=int, default=1,
        help='The depth of an EncDec pair in the SegNet model to be trained.'
             'This means the number of inside pairs of Encoder-Decoder.')
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
        '--n_encdec', type=int, default=4,
        help='The number of Encoder-Decoder pairs that are included in the'
             ' Segnet model')
    parser.add_argument(
        '--in_channel', type=int, default=3,
        help='The number of channels of the input image')
    parser.add_argument(
        '--n_mid', type=int, default=64,
        help='The number of channels of convolutions in EncDec unit')
    parser.add_argument(
        '--mean', type=str, default=None,
        help='Mean npy over the training data')
    parser.add_argument(
        '--std', type=str, default=None,
        help='Stddev npy over the training data')
    parser.add_argument(
        '--use_class_weight', action='store_true', default=False,
        help='If it\'s given, the loss is weighted during training.')
    parser.add_argument(
        '--class_weight', type=str, default='data/train_freq.csv',
        help='The path to the file that contains inverse class frequency '
             'calculated using calc_mean.py')
    parser.add_argument(
        '--ignore_labels', type=str, default='11',
        help='The label id that is ignored during training')
    parser.add_argument(
        '--finetune', action='store_true', default=False,
        help='Train whole encoder-decorder pairs at a time.')

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
        '--opt', type=str, default='MomentumSGD',
        choices=['MomentumSGD', 'Adam', 'AdaGrad', 'RMSprop'],
        help='Optimization method')
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--adam_alpha', type=float, default=0.001)
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

    parser.add_argument('-f')  # To call this from jupyter notebook

    args = parser.parse_args()
    args.class_weight = [
        float(w) for w in open(args.class_weight).readline().split(',')]

    xp = chainer.cuda.cupy if chainer.cuda.available else numpy
    random.seed(args.seed)
    xp.random.seed(args.seed)
    numpy.random.seed(args.seed)
    args.ignore_labels = [int(l) for l in args.ignore_labels.split(',')]

    return args
