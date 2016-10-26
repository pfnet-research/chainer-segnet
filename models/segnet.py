#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2016 Shunta Saito

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from chainer import reporter
from models.weighted_softmax_cross_entropy import \
    weighted_softmax_cross_entropy

import chainer
import chainer.functions as F
import chainer.links as L
import math
import numpy as np


class SegNet(chainer.Chain):

    def __init__(self, n_classes):
        w = math.sqrt(2)
        super(SegNet, self).__init__(
            conv1=L.Convolution2D(3, 64, 7, 1, 3, w),
            bn1=L.BatchNormalization(64),
            conv2=L.Convolution2D(64, 64, 7, 1, 3, w),
            bn2=L.BatchNormalization(64),
            conv3=L.Convolution2D(64, 64, 7, 1, 3, w),
            bn3=L.BatchNormalization(64),
            conv4=L.Convolution2D(64, 64, 7, 1, 3, w),
            bn4=L.BatchNormalization(64),
            conv5=L.Convolution2D(64, 64, 7, 1, 3, w),
            bn5=L.BatchNormalization(64),
            conv6=L.Convolution2D(64, 64, 7, 1, 3, w),
            bn6=L.BatchNormalization(64),
            conv7=L.Convolution2D(64, 64, 7, 1, 3, w),
            bn7=L.BatchNormalization(64),
            conv8=L.Convolution2D(64, 64, 7, 1, 3, w),
            bn8=L.BatchNormalization(64),
            conv_cls=L.Convolution2D(64, n_classes, 1, 1, 0, w)
        )
        self.n_classes = n_classes
        self.pools = [F.MaxPooling2D(2, 2) for _ in range(4)]
        self.train = True

    def __call__(self, x):
        h = F.local_response_normalization(x, 5, 1, 0.0001, 0.75)
        h0, w0 = h.shape[2:]
        h = self.pools[0](F.relu(self.bn1(self.conv1(h), test=not self.train)))
        h1, w1 = h.shape[2:]
        h = self.pools[1](F.relu(self.bn2(self.conv2(h), test=not self.train)))
        h2, w2 = h.shape[2:]
        h = self.pools[2](F.relu(self.bn3(self.conv3(h), test=not self.train)))
        h3, w3 = h.shape[2:]
        h = self.pools[3](F.relu(self.bn4(self.conv4(h), test=not self.train)))
        h = F.unpooling_2d(h, 2, 2, outsize=(h3, w3))
        h = self.bn5(self.conv5(h), test=not self.train)
        h = F.unpooling_2d(h, 2, 2, outsize=(h2, w2))
        h = self.bn6(self.conv6(h), test=not self.train)
        h = F.unpooling_2d(h, 2, 2, outsize=(h1, w1))
        h = self.bn7(self.conv7(h), test=not self.train)
        h = F.unpooling_2d(h, 2, 2, outsize=(h0, w0))
        return self.conv_cls(self.bn8(self.conv8(h), test=not self.train))


class SegNetLoss(chainer.Chain):

    def __init__(self, model, class_weights=None):
        super(SegNetLoss, self).__init__(predictor=model)
        if class_weights is not None:
            class_weights = np.asarray([class_weights], dtype=np.float32)
            class_weights = chainer.Variable(class_weights, 'auto')
            class_weights = F.expand_dims(class_weights, 2)
            self.class_weights = F.expand_dims(class_weights, 3)
            assert(self.class_weights.shape == (1, model.n_classes, 1, 1))

    def __call__(self, x, t):
        y = self.predictor(x)
        if hasattr(self, 'class_weights'):
            w = F.broadcast_to(self.class_weights, y.shape)
            with chainer.cuda.get_device(x.data) as gid:
                w.data = chainer.cuda.to_gpu(w.data, int(gid))
            self.loss = weighted_softmax_cross_entropy(y, t, w)
        else:
            self.loss = F.softmax_cross_entropy(y, t)
        reporter.report({'loss': self.loss}, self)
        return self.loss
