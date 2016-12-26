#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import cv2 as cv
import glob
import numpy as np
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_dir', type=str)
    parser.add_argument('--anno_dir', type=str, default='data/testannot')
    parser.add_argument('--n_classes', type=int, default=12)
    parser.add_argument('--unknown_class', type=int, default=11)
    args = parser.parse_args()

    anno_image_fns = sorted(glob.glob('{}/*.png'.format(args.anno_dir)))
    result_npy_fns = sorted(glob.glob('{}/*.npy'.format(args.result_dir)))

    print('anno_imag_fns  n = {}'.format(len(anno_image_fns)))
    print('result_npy_fns n = {}'.format(len(result_npy_fns)))

    totalpoints = 0
    cf = np.zeros((len(anno_image_fns), args.n_classes, args.n_classes))
    globalacc = 0

    for i, (anno_img_fn, result_npy_fn) in enumerate(
            zip(anno_image_fns, result_npy_fns)):
        assert os.path.splitext(os.path.basename(anno_img_fn))[0] == \
            os.path.splitext(os.path.basename(result_npy_fn))[0]
        anno = cv.imread(anno_img_fn, cv.IMREAD_GRAYSCALE)
        pred = np.load(result_npy_fn)

        totalpoints += np.sum(anno != args.unknown_class)

        for j in range(args.n_classes):
            if j == args.unknown_class:
                continue
            c1 = anno == j
            for k in range(args.n_classes):
                c1p = pred == k
                index = c1 * c1p
                cf[i, j, k] += np.sum(index)
                if j == k:
                    globalacc += np.sum(index)

cf = np.sum(cf, axis=0)

# Compute confusion matrix
conf = np.zeros((args.n_classes, args.n_classes))
for i in range(args.n_classes):
    if i != args.unknown_class and cf[i, :].sum() > 0:
        conf[i, :] = cf[i, :] / cf[i, :].sum()
globalacc /= float(totalpoints)

# Compute intersection over union for each class and its mean
iou = np.zeros((args.n_classes,))
for i in range(args.n_classes):
    if i != args.unknown_class and conf[i, :].sum() > 0:
        iou[i] = cf[i, i] / cf[i, :].sum() + cf[:, i].sum() - cf[i, i]

print('Global acc = {}'.format(globalacc))
print('Class average acc = {}'.format(
    np.diag(conf).sum() / float(args.n_classes)))
print('Mean IoU = {}'.format(iou.sum() / float(args.n_classes)))
