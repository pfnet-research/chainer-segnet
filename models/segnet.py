# Copyright (c) 2016 Shunta Saito

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from chainer import reporter
from models.upsampling_2d import upsampling_2d

import chainer
import chainer.functions as F
import chainer.links as L
import math
import models.weighted_softmax_cross_entropy as wsce
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
        self.pools = [F.MaxPooling2D(2, 2, use_cudnn=False) for _ in range(4)]
        self.train = True

    def upsampling_2d(self, pooler, x, outsize):
        print(outsize)
        return upsampling_2d(
            x, pooler.indexes, ksize=(pooler.kh, pooler.kw),
            stride=(pooler.sy, pooler.sx), pad=(pooler.ph, pooler.pw),
            outsize=outsize)

    def __call__(self, x):
        # Encoder
        h = F.local_response_normalization(x, 5, 1, 0.0005, 0.75)
        h0, w0 = h.shape[2:]
        h = self.pools[0](F.relu(self.bn1(self.conv1(h), test=not self.train)))
        h1, w1 = h.shape[2:]
        h = self.pools[1](F.relu(self.bn2(self.conv2(h), test=not self.train)))
        h2, w2 = h.shape[2:]
        h = self.pools[2](F.relu(self.bn3(self.conv3(h), test=not self.train)))
        h3, w3 = h.shape[2:]
        h = self.pools[3](F.relu(self.bn4(self.conv4(h), test=not self.train)))

        # Decoder
        print('--indexes--')
        print(self.pools[3].indexes.shape)
        print(self.pools[2].indexes.shape)
        print(self.pools[1].indexes.shape)
        print(self.pools[0].indexes.shape)
        print('-----------')
        h = self.upsampling_2d(self.pools[3], h, (h3, w3))
        h = self.bn5(self.conv5(h), test=not self.train)
        print(h.shape)
        h = self.upsampling_2d(self.pools[2], h, (h2, w2))
        h = self.bn6(self.conv6(h), test=not self.train)
        print(h.shape)
        h = self.upsampling_2d(self.pools[1], h, (h1, w1))
        h = self.bn7(self.conv7(h), test=not self.train)
        print(h.shape)
        h = self.upsampling_2d(self.pools[0], h, (h0, w0))
        print(h.shape)
        h = self.conv_cls(self.bn8(self.conv8(h), test=not self.train))
        print(h.shape)
        return h


class SegNetLoss(chainer.Chain):

    def __init__(self, model, class_weights=None):
        super(SegNetLoss, self).__init__(predictor=model)
        if class_weights is not None:
            if not isinstance(class_weights, np.ndarray):
                class_weights = np.asarray(class_weights, dtype=np.float32)
            self.class_weights = class_weights
            assert self.class_weights.ndim == 1
            assert len(self.class_weights) == model.n_classes

    def __call__(self, x, t):
        y = self.predictor(x)
        if hasattr(self, 'class_weights'):
            self.loss = wsce.weighted_softmax_cross_entropy(
                y, t, self.class_weights)
        else:
            self.loss = F.softmax_cross_entropy(y, t)
        reporter.report({'loss': self.loss}, self)
        return self.loss
