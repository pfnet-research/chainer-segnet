#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2016 Shunta Saito

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from lib import CamVid

import os
import cv2 as cv

if __name__ == '__main__':
    img_dir = 'data/train'
    lbl_dir = 'data/trainannot'
    list_fn = 'data/train.txt'
    mean = None
    std = None
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
    for i in range(100):
        img, lbl = camvid[i]
        img = img.transpose(1, 2, 0)
        cv.imwrite('{}/{}_img.png'.format(out_dir, i), img)
        cv.imwrite('{}/{}_lbl.png'.format(out_dir, i), lbl * 20)
        print(img.shape, lbl.shape)
