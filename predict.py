#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2016 Shunta Saito

import argparse
import chainer
import chainer.functions as F
import cv2 as cv
import glob
import imp
import json
import numpy as np
import os
import six

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
    parser.add_argument('--train_depth', type=int)
    parser.add_argument('--saved_args', type=str, default='results/args.json')
    parser.add_argument('--snapshot', type=str)
    parser.add_argument('--test_img_dir', type=str, default='data/test')
    parser.add_argument('--out_dir', type=str, default='results/prediction')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--mean', type=str, default=None)
    parser.add_argument('--std', type=str, default=None)
    parser.add_argument('--scale', type=float, default=1.0)
    parser.add_argument('--save_output', action='store_true', default=False)
    args = parser.parse_args()

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    loaded_args = json.load(open(args.saved_args))
    model = imp.load_source(
        loaded_args['model_name'], loaded_args['model_file'])
    model = getattr(model, loaded_args['model_name'])(
        loaded_args['n_encdec'], loaded_args['n_classes'],
        loaded_args['in_channel'], loaded_args['n_mid'])

    # Load parameters
    param = np.load(args.snapshot)
    prefix = 'updater/model:main/predictor'
    for key, arr in six.iteritems(param):
        if prefix in key:
            key = key.replace(prefix, '')
            names = [k for k in key.split('/') if len(k) > 0]
            link = model
            for name in names[:-1]:
                link = link.__dict__[name]
            if isinstance(link.__dict__[names[-1]], chainer.Variable):
                link.__dict__[names[-1]] = chainer.Variable(arr, volatile='on')
            else:
                link.__dict__[names[-1]] = arr

    if args.gpu >= 0:
        model.to_gpu(args.gpu)
        print('Model is transferred to GPU: {}'.format(args.gpu))
    model.train = False

    mean = None if args.mean is None else np.load(args.mean)
    std = None if args.std is None else np.load(args.std)

    print('mean = {}, std = {}'.format(mean, std))

    for img_fn in sorted(glob.glob('{}/*.png'.format(args.test_img_dir))):
        # Load & prepare image
        img = cv.imread(img_fn).astype(np.float)
        if mean is not None:
            img -= mean
        if std is not None:
            img /= std
        if mean is None and std is None:
            img /= 255.0
        if args.scale != 1.0:
            img = cv.resize(img, None, fx=args.scale, fy=args.scale)
        img = np.expand_dims(img, 0).transpose(0, 3, 1, 2)
        img = img.astype(np.float32)
        if args.gpu >= 0:
            with chainer.cuda.Device(args.gpu):
                img = chainer.cuda.cupy.asarray(img)
        img_var = chainer.Variable(img, volatile='on')

        # Forward
        ret = model(img_var, depth=args.train_depth)
        ret = F.softmax(ret).data[0].transpose(1, 2, 0)
        if args.gpu >= 0:
            with chainer.cuda.Device(args.gpu):
                ret = chainer.cuda.to_cpu(ret)
        if args.save_output:
            np.save('{}/{}_full'.format(
                args.out_dir,
                os.path.splitext(os.path.basename(img_fn))[0]), ret)

        # Create output
        out = np.zeros((ret.shape[0], ret.shape[1], 3), dtype=np.uint8)
        mask = np.argmax(ret, axis=2)
        for y in range(mask.shape[0]):
            for x in range(mask.shape[1]):
                out[y, x] = colors[mask[y, x]]

        base_fn = os.path.basename(img_fn)
        out_fn = '{}/{}'.format(args.out_dir, base_fn)
        cv.imwrite(out_fn, out)
        out_fn = '{}/{}'.format(
            args.out_dir,
            os.path.splitext(os.path.basename(out_fn))[0])
        np.save(out_fn, mask)
        print(img_fn, img.shape, out_fn)
