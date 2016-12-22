# Copyright (c) 2016 Shunta Saito

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from chainer.dataset import dataset_mixin
from os.path import basename as bn

import cv2 as cv
import numpy as np
import os
import re


def _get_img_id(fn):
    bname = os.path.basename(fn)
    return re.search('([a-zA-Z\-]+_[0-9]+_[0-9]+)', bname).groups()[0]


class CamVid(dataset_mixin.DatasetMixin):

    def __init__(self, img_dir, lbl_dir, list_fn, mean, std, shift_jitter=0,
                 scale_jitter=0, fliplr=False, rotate=None, rotate_max=7,
                 scale=1.0, ignore_labels=[11]):
        self.scale = scale
        self.lbl_fns = []
        self.img_fns = []
        for line in open(list_fn):
            img_fn, lbl_fn = line.split()
            self.img_fns.append('{}/{}'.format(img_dir, bn(img_fn)))
            self.lbl_fns.append('{}/{}'.format(lbl_dir, bn(lbl_fn)))
        self.mean = None if mean is None else np.load(mean)
        self.std = None if std is None else np.load(std)
        self.shift_jitter = shift_jitter
        self.scale_jitter = scale_jitter
        self.fliplr = fliplr
        self.rotate = rotate
        self.rotate_max = rotate_max
        self.ignore_labels = ignore_labels

    def __len__(self):
        return len(self.img_fns)

    def get_example(self, i):
        img = cv.imread(self.img_fns[i]).astype(np.float)
        if self.mean is not None:
            img -= self.mean
        if self.std is not None:
            img /= self.std
        if self.mean is None and self.std is None:
            img /= 255.0
        if self.scale != 1.0:
            img = cv.resize(img, None, fx=self.scale, fy=self.scale,
                            interpolation=cv.INTER_NEAREST)

        lbl = cv.imread(self.lbl_fns[i], cv.IMREAD_GRAYSCALE).astype(np.int32)
        if self.scale != 1.0:
            lbl = cv.resize(lbl, None, fx=self.scale, fy=self.scale,
                            interpolation=cv.INTER_NEAREST)
        if self.ignore_labels is not None:
            for ignore_l in self.ignore_labels:
                lbl[np.where(lbl == ignore_l)] = -1
        img_size = (lbl.shape[1], lbl.shape[0])  # W, H

        if self.shift_jitter != 0:
            s = np.random.randint(-self.shift_jitter, self.shift_jitter)
            cy, cx = img.shape[0] // 2 + s, img.shape[1] // 2 + s
            hh = (img.shape[0] - 2 * self.shift_jitter) // 2
            hw = (img.shape[1] - 2 * self.shift_jitter) // 2
            img = img[cy - hh:cy + hh, cx - hw:cx + hw]
            lbl = lbl[cy - hh:cy + hh, cx - hw:cx + hw]
            img = cv.resize(img, img_size, interpolation=cv.INTER_NEAREST)
            lbl = cv.resize(lbl, img_size, interpolation=cv.INTER_NEAREST)

        if self.scale_jitter != 0.0:
            h, w = img.shape[:2]
            s = np.random.uniform(1 - self.scale_jitter, 1 + self.scale_jitter)
            img = cv.resize(img, None, fx=s, fy=s,
                            interpolation=cv.INTER_NEAREST)
            lbl = cv.resize(lbl, None, fx=s, fy=s,
                            interpolation=cv.INTER_NEAREST)
            img = cv.resize(img, img_size, interpolation=cv.INTER_NEAREST)
            lbl = cv.resize(lbl, img_size, interpolation=cv.INTER_NEAREST)

        if self.rotate:
            h, w = img.shape[:2]
            s = np.clip(np.random.normal(), -self.rotate_max, self.rotate_max)
            mat = cv.getRotationMatrix2D((w // 2, h // 2), s, 1)
            img = cv.warpAffine(img, mat, (w, h), flags=cv.INTER_NEAREST)
            lbl = cv.warpAffine(lbl.astype(np.float), mat,
                                (w, h), flags=cv.INTER_NEAREST, borderValue=-1)
            lbl = lbl.astype(np.int32)

        if self.fliplr and np.random.randint(0, 2) == 1:
            img = cv.flip(img, 1)
            lbl = cv.flip(lbl, 1)

        lbl = lbl.astype(np.int32)
        img = img.transpose(2, 0, 1).astype(np.float32)

        return img, lbl
