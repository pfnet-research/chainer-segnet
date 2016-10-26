#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2016 Shunta Saito

import argparse
import chainer
import chainer.functions as F
import cv2 as cv
import glob
import imp
import numpy as np
import os
import six

colors = [
    [64, 128, 64],  # Animal
    [192, 0, 128],  # Archway
    [0, 128, 192],  # Bicyclist
    [0, 128, 64],  # Bridge
    [128, 0, 0],  # Building
    [64, 0, 128],  # Car
    [64, 0, 192],  # CartLuggagePram
    [192, 128, 64],  # Child
    [192, 192, 128],  # Column_Pole
    [64, 64, 128],  # Fence
    [128, 0, 192],  # LaneMkgsDriv
    [192, 0, 64],  # LaneMkgsNonDriv
    [128, 128, 64],  # Misc_Text
    [192, 0, 192],  # MotorcycleScooter
    [128, 64, 64],  # OtherMoving
    [64, 192, 128],  # ParkingBlock
    [64, 64, 0],  # Pedestrian
    [128, 64, 128],  # Road
    [128, 128, 192],  # RoadShoulder
    [0, 0, 192],  # Sidewalk
    [192, 128, 128],  # SignSymbol
    [128, 128, 128],  # Sky
    [64, 128, 192],  # SUVPickupTruck
    [0, 0, 64],  # TrafficCone
    [0, 64, 64],  # TrafficLight
    [192, 64, 128],  # Train
    [128, 128, 0],  # Tree
    [192, 128, 192],  # Truck_Bus
    [64, 0, 64],  # Tunnel
    [192, 192, 0],  # VegetationMisc
    [0, 0, 0],  # Void
    [64, 192, 0],  # Wall
]
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_file', type=str, default='models/segnet.py')
    parser.add_argument('--model_name', type=str, default='SegNet')
    parser.add_argument('--n_classes', type=int, default=11)
    parser.add_argument('--snapshot', type=str)
    parser.add_argument('--img_dir', type=str, default='data/test')
    parser.add_argument('--out_dir', type=str)
    parser.add_argument('--scale', type=float, default=1.0)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--mean', type=str, default=None)
    parser.add_argument('--std', type=str, default=None)
    args = parser.parse_args()

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    model = imp.load_source(args.model_name, args.model_file)
    model = getattr(model, args.model_name)(args.n_classes)
    param = np.load(args.snapshot)
    prefix = 'updater/model:main/predictor'

    # Load parameters
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
    model.train = False

    mean = None if args.mean is None else np.load(args.mean)
    std = None if args.std is None else np.load(args.std)

    for img_fn in sorted(glob.glob('{}/*.png'.format(args.img_dir))):
        print(img_fn)

        # Load & prepare image
        img = cv.imread(img_fn).astype(np.float)
        if mean is not None:
            img -= mean
        if std is not None:
            img /= std
        if args.scale != 1.0:
            img = cv.resize(img, None, fx=args.scale, fy=args.scale)
        img = np.expand_dims(img, 0).transpose(0, 3, 1, 2)
        img = img.astype(np.float32)
        if args.gpu >= 0:
            with chainer.cuda.Device(args.gpu):
                img = chainer.cuda.cupy.asarray(img)
        img_var = chainer.Variable(img, volatile='on')

        # Forward
        ret = model(img_var)
        ret = F.softmax(ret).data[0].transpose(1, 2, 0)
        if args.gpu >= 0:
            with chainer.cuda.Device(args.gpu):
                ret = chainer.cuda.cupy.asnumpy(ret)

        # Create output
        out = np.zeros((ret.shape[0], ret.shape[1], 3), dtype=np.uint8)
        mask = np.argmax(ret, axis=2)
        for y in range(mask.shape[0]):
            for x in range(mask.shape[1]):
                out[y, x] = colors[mask[y, x]]

        base_fn = os.path.basename(img_fn)
        out_fn = '{}/{}'.format(args.out_dir, base_fn)
        cv.imwrite(out_fn, out)
        print(out_fn)
