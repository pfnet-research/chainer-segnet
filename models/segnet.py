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
import logging
import math
import models.weighted_softmax_cross_entropy as wsce
import numpy as np
import six


class EncDec(chainer.Chain):

    def __init__(self, in_channel, n_mid=64):
        w = math.sqrt(2)
        super(EncDec, self).__init__(
            enc=L.Convolution2D(in_channel, n_mid, 7, 1, 3, w),
            bn_m=L.BatchNormalization(n_mid),
            dec=L.Convolution2D(n_mid, n_mid, 7, 1, 3, w),
        )
        self.p = F.MaxPooling2D(2, 2, use_cudnn=False)
        self.inside = None

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
        h = self.dec(self.upsampling_2d(self.p, h, x.shape[2:]))
        if out_bn:
            if not hasattr(self, 'bn_o'):
                self.add_link('bn_o', L.BatchNormalization(64))
                if isinstance(x.data, cuda.cupy.ndarray):
                    self.bn_o.to_gpu(x.data.device)
            h = self.bn_o(h, test=not train)
        return h


class SegNet(chainer.Chain):

    """SegNet model architecture.

    This class has all Links to construct SegNet model and also optimizers to
    optimize each part (Encoder-Decoder pair or conv_cls) of SegNet.
    """

    def __init__(self, optimizer=None, n_encdec=4, n_classes=12, n_mid=64):
        assert n_encdec >= 1
        w = math.sqrt(2)
        super(SegNet, self).__init__(
            conv_cls=L.Convolution2D(n_mid, n_classes, 1, 1, 0, w))

        # Create and add EncDecs
        names = ['conv_cls']
        for i in six.moves.range(1, n_encdec + 1):
            name = 'encdec{}'.format(i)
            encdec = EncDec(n_mid if i > 1 else 3, n_mid)
            self.add_link(name, encdec)
            names.append(name)

        # Setup each optimizer for each EncDec or conv_cls
        if optimizer is not None:
            self.optimizers = {}
            for name in names:
                opt = copy.deepcopy(optimizer)
                opt.setup(getattr(self, name))
                self.optimizers[name] = opt

        # Add WeightDecay if it's specified by 'args'
        for name, opt in self.optimizers.items():
            if hasattr(opt, 'decay') and 'WeightDecay' not in opt._hooks:
                opt.add_hook(chainer.optimizer.WeightDecay(opt.decay))

        self.n_encdec = n_encdec
        self.n_classes = n_classes
        self.train = True

    def __call__(self, x, depth=1):
        assert 1 <= depth <= self.n_encdec

        h = F.local_response_normalization(x, 5, 1, 0.0005, 0.75)

        for d in six.moves.range(1, depth + 1):
            encdec = getattr(self, 'encdec{}'.format(d))
            encdec.inside = None
        for d in six.moves.range(1, depth + 1):
            encdec = getattr(self, 'encdec{}'.format(d))
            if depth >= d + 1:
                encdec.inside = getattr(self, 'encdec{}'.format(d + 1))
        h = self.encdec1(h, train=self.train)
        h = self.conv_cls(h)

        if self.train:
            name = 'encdec{}'.format(depth)
            optimizers = [(name, self.optimizers[name])]
            if depth == 1:
                # Optimize conv_cls only during training for encdec1
                optimizers.append(('conv_cls', self.optimizers['conv_cls']))
            # Number of EncDec trained at a time is always one
            assert 1 <= len(optimizers) <= 2
            return h, dict(optimizers)
        else:
            return h


class SegNetLoss(chainer.Chain):

    def __init__(self, model, class_weights=None):
        super(SegNetLoss, self).__init__(predictor=model)
        if class_weights is not None:
            logging.info('class_weights: {}'.format(class_weights))
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
