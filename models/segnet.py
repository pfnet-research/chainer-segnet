# Copyright (c) 2016 Shunta Saito

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from chainer import cuda
from chainer import reporter
from models.softmax_cross_entropy import softmax_cross_entropy
from models.upsampling_2d import upsampling_2d

import chainer
import chainer.functions as F
import chainer.links as L
import logging
import math
import numpy as np
import six


class EncDec(chainer.Chain):

    def __init__(self, in_channel, n_mid=64):
        w = math.sqrt(2)
        super(EncDec, self).__init__(
            enc=L.Convolution2D(in_channel, n_mid, 7, 1, 3, w),
            bn_m=L.BatchNormalization(n_mid),
            dec=L.Convolution2D(n_mid, n_mid, 7, 1, 3, w),
            bn_o=L.BatchNormalization(n_mid),
        )
        self.p = F.MaxPooling2D(2, 2, use_cudnn=False)
        self.inside = None

    def upsampling_2d(self, pooler, x, outsize):
        return upsampling_2d(
            x, pooler.indexes, ksize=(pooler.kh, pooler.kw),
            stride=(pooler.sy, pooler.sx), pad=(pooler.ph, pooler.pw),
            outsize=outsize)

    def __call__(self, x, train=True):
        # Encode
        h = self.p(F.relu(self.bn_m(self.enc(x), test=not train)))

        # Run the inside network
        if self.inside is not None:
            h = self.inside(h, train)

        # Decode
        h = self.dec(self.upsampling_2d(self.p, h, x.shape[2:]))
        return self.bn_o(h, test=not train)


class SegNet(chainer.Chain):

    """SegNet model architecture.

    This class has all Links to construct SegNet model and also optimizers to
    optimize each part (Encoder-Decoder pair or conv_cls) of SegNet.
    """

    def __init__(self, n_encdec=4, n_classes=12, in_channel=3, n_mid=64):
        assert n_encdec >= 1
        w = math.sqrt(2)
        super(SegNet, self).__init__(
            conv_cls=L.Convolution2D(n_mid, n_classes, 1, 1, 0, w))

        # Create and add EncDecs
        for i in six.moves.range(1, n_encdec + 1):
            name = 'encdec{}'.format(i)
            self.add_link(name, EncDec(n_mid if i > 1 else in_channel, n_mid))
        for d in six.moves.range(1, n_encdec):
            encdec = getattr(self, 'encdec{}'.format(d))
            encdec.inside = getattr(self, 'encdec{}'.format(d + 1))
            setattr(self, 'encdec{}'.format(d), encdec)

        self.n_encdec = n_encdec
        self.n_classes = n_classes
        self.train = True

    def is_registered_link(self, name):
        return name in self._children

    def remove_link(self, name):
        """Remove a link that has the given name from this model

        Optimizer sees ``~Chain.namedparams()`` to know which parameters should
        be updated. And inside of ``namedparams()``, ``self._children`` is
        called to get names of all links included in the Chain.
        """
        self._children.remove(name)

    def recover_link(self, name):
        self._children.append(name)

    def __call__(self, x, depth=1):
        assert 1 <= depth <= self.n_encdec
        h = F.local_response_normalization(x, 5, 1, 0.0005, 0.75)

        # Unchain the inner EncDecs after the given depth
        encdec = getattr(self, 'encdec{}'.format(depth))
        encdec.inside = None

        h = self.encdec1(h, train=self.train)
        h = self.conv_cls(h)
        return h


class SegNetLoss(chainer.Chain):

    def __init__(self, model, class_weight=None, train_depth=1):
        super(SegNetLoss, self).__init__(predictor=model)
        if class_weight is not None:
            logging.info('class_weight: {}'.format(class_weight))
            if not isinstance(class_weight, np.ndarray):
                class_weight = np.asarray(class_weight, dtype=np.float32)
            self.class_weight = class_weight
            assert len(self.class_weight) == model.n_classes
        self.train_depth = train_depth

    def __call__(self, x, t):
        self.y = self.predictor(x, self.train_depth)
        if hasattr(self, 'class_weight'):
            if isinstance(x.data, cuda.cupy.ndarray) \
                    and not isinstance(self.class_weight, cuda.cupy.ndarray):
                self.class_weight = cuda.to_gpu(
                    self.class_weight, device=x.data.device)
            self.loss = softmax_cross_entropy(
                self.y, t, class_weight=self.class_weight)
        else:
            self.loss = F.softmax_cross_entropy(self.y, t)
        reporter.report({'loss': self.loss}, self)
        return self.loss
