# Copyright (c) 2016 Shunta Saito

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from chainer import cuda
from chainer import reporter
from models.upsampling_2d import upsampling_2d

import chainer
import chainer.functions as F
import chainer.links as L
import copy
import math
import models.weighted_softmax_cross_entropy as wsce
import numpy as np


class EncDec(chainer.Chain):

    def __init__(self, inside=None):
        w = math.sqrt(2)
        super(EncDec, self).__init__(
            enc=L.Convolution2D(3, 64, 7, 1, 3, w),
            bn_m=L.BatchNormalization(64),
            dec=L.Convolution2D(3, 64, 7, 1, 3, w),
        )
        self.p = F.MaxPooling2D(2, 2, use_cudnn=False)
        self.inside = inside

    def upsampling_2d(self, pooler, x, outsize):
        return upsampling_2d(
            x, pooler.indexes, ksize=(pooler.kh, pooler.kw),
            stride=(pooler.sy, pooler.sx), pad=(pooler.ph, pooler.pw),
            outsize=outsize)

    def __call__(self, x, out_bn=True, train=True):
        # Encode
        h = self.p(F.relu(self.bn_m(self.enc(x), test=not train)))

        # Run the inside network
        if self.inside is not None:
            h = self.inside(h)

        # Decode
        h = self.upsampling_2d(self.p, h, x.shape[2:])
        if out_bn:
            if not hasattr(self, 'bn_o'):
                self.add_link('bn_o', L.BatchNormalization(64))
            h = self.bn_o(h, test=not train)
        return h


class SegNet(chainer.Chain):

    def __init__(self, optimizer, n_encdec=4, n_classes=12):
        w = math.sqrt(2)
        super(SegNet, self).__init__(
            conv_cls=L.Convolution2D(64, n_classes, 1, 1, 0, w)
        )

        # Setup each optimizer for each encdec
        self.optimizers = {}
        for i in range(n_encdec):
            name = 'encdec{}'.format(i)
            if i == 0:
                encdec = EncDec(inside=None)
            else:
                inside = getattr(self, 'encdec{}'.format(i - 1))
                encdec = EncDec(inside=inside)
            self.add_link(name, encdec)
            opt = copy.deepcopy(optimizer)
            opt.setup(getattr(self, name))
            self.optimizers[name] = opt
        opt = copy.deepcopy(optimizer)
        opt.setup(self.conv_cls)
        self.optimizers['conv_cls'] = opt

        self.n_encdec = n_encdec
        self.n_classes = n_classes
        self.train = True

    def __call__(self, x, depth=1):
        assert depth < self.n_encdec

        h = F.local_response_normalization(x, 5, 1, 0.0005, 0.75)
        name = 'encdec{}'.format(depth - 1)
        h = getattr(self, name)(h, train=self.train)
        h = self.conv_cls(self.bn8(self.conv8(h), test=not self.train))

        optimizers = [self.optimizers[name]]
        if depth == 1:
            # Optimize the output conv layer only during first stage training
            optimizers.append(self.optimizers['conv_cls'])

        return h, optimizers


class SegNetLoss(chainer.Chain):

    def __init__(self, model, class_weights=None):
        super(SegNetLoss, self).__init__(predictor=model)
        if class_weights is not None:
            if not isinstance(class_weights, np.ndarray):
                class_weights = np.asarray(class_weights, dtype=np.float32)
            self.class_weights = class_weights
            assert self.class_weights.ndim == 1
            assert len(self.class_weights) == model.n_classes

    def __call__(self, x, t, depth=1):
        y, optimizers = self.predictor(x, depth)
        if hasattr(self, 'class_weights'):
            if isinstance(x.data, cuda.cupy.ndarray):
                self.class_weights = cuda.to_gpu(
                    self.class_weights, device=x.data.device)
            self.loss = wsce.weighted_softmax_cross_entropy(
                y, t, self.class_weights)
        else:
            self.loss = F.softmax_cross_entropy(y, t)
        reporter.report({'loss': self.loss}, self)
        return self.loss, optimizers
