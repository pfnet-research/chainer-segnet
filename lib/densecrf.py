#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import cv2 as cv
import glob
import numpy as np
import os
import pydensecrf.densecrf as dcrf

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
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_dir', type=str)
    args = parser.parse_args()

    for fn in glob.glob('{}/*_full.npy'.format(args.result_dir)):
        img_fn = 'data/test/' + os.path.splitext(
            os.path.basename(fn))[0].replace('_full', '.png')
        im = cv.imread(img_fn).transpose(1, 0, 2)
        im = np.ascontiguousarray(im)

        lbl_fn = img_fn.replace('test', 'testannot')
        lbl = cv.imread(lbl_fn, cv.IMREAD_GRAYSCALE)
        annot = np.zeros((lbl.shape[0], lbl.shape[1], 3), dtype=np.uint8)
        for k in range(12):
            annot[np.where(lbl == k)] = colors[k]
        lbl_fn = '{}/{}_annot.png'.format(
            args.result_dir, os.path.splitext(os.path.basename(fn))[0])
        cv.imwrite(lbl_fn, annot)

        unary = -np.log(np.load(fn))
        unary = unary.transpose(1, 0, 2)
        width, height, nlabels = unary.shape
        unary = unary.transpose(2, 0, 1).reshape(nlabels, -1)
        unary = np.ascontiguousarray(unary)

        d = dcrf.DenseCRF2D(width, height, nlabels)
        d.setUnaryEnergy(unary)
        d.addPairwiseBilateral(sxy=5, srgb=3, rgbim=im, compat=5)

        q = d.inference(5)
        mask = np.argmax(q, axis=0).reshape(width, height).transpose(1, 0)
        print(np.unique(mask))

        out_lbl = np.zeros((height, width, 3), dtype=np.uint8)
        for k in range(12):
            out_lbl[np.where(mask == k)] = colors[k]
        fn = '{}/{}_densecrf.png'.format(
            args.result_dir, os.path.splitext(os.path.basename(fn))[0])
        cv.imwrite(fn, out_lbl)
        break
