#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2016 Shunta Saito

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import cv2 as cv
import numpy as np
import os
import sys

if True:
    sys.path.insert(0, '.')
    from lib import CamVid

Sky = [128, 128, 128]
Building = [128, 0, 0]
Pole = [192, 192, 128]
Road_marking = [255, 69, 0]
Road = [128, 64, 128]
Pavement = [60, 40, 222]
Tree = [128, 128, 0]
SignSymbol = [192, 128, 128]
Fence = [64, 64, 128]
Car = [64, 0, 128]
Pedestrian = [64, 64, 0]
Bicyclist = [0, 128, 192]
Unlabelled = [0, 0, 0]

colors = [Sky, Building, Pole, Road, Pavement, Tree, SignSymbol, Fence,
          Car, Pedestrian, Bicyclist, Unlabelled]


if __name__ == '__main__':
    img_dir = 'data/train'
    lbl_dir = 'data/trainannot'
    list_fn = 'data/train.txt'
    mean = 'data/train_mean.npy'
    std = 'data/train_std.npy'
    shift_jitter = 50
    scale_jitter = 0.2
    fliplr = True
    rotate = True
    rotate_max = 7
    scale = 1.0
    camvid = CamVid(img_dir, lbl_dir, list_fn, mean, std, shift_jitter,
                    scale_jitter, fliplr, rotate, rotate_max, scale)

    out_dir = 'tests/out'
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    for i in range(len(camvid)):
        img, lbl = camvid[i]
        lbl_ids = np.unique(lbl)
        img = img.transpose(1, 2, 0)
        print(img.shape, lbl.shape, lbl_ids)
        img -= img.reshape(-1, 3).min(axis=0)
        img /= img.reshape(-1, 3).max(axis=0)
        img *= 255
        cv.imwrite('{}/{}_img.png'.format(out_dir, i), img)
        out_lbl = np.zeros_like(img)
        for k in range(12):
            out_lbl[np.where(lbl == k)] = colors[k]
        cv.imwrite('{}/{}_lbl.png'.format(out_dir, i), out_lbl)
        assert len(lbl_ids) <= 12
        assert lbl_ids.min() >= -1
        assert lbl_ids.max() <= 11
