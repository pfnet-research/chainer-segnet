# Copyright (c) 2016 Shunta Saito

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from chainer import cuda
from chainer.utils import conv
from chainer.utils import type_check

import chainer
import itertools
import numpy as np


class Upsampling2D(chainer.Function):

    """Upsampling over a set of 2d planes w/ indices used for max pooling."""

    def __init__(self, pooler, outsize):
        self.p = pooler
        self.outsize = outsize

    def forward_cpu(self, x):
        n, c, h, w, oh, ow = x[0].shape + self.outsize
        print(n, c, h, w, oh, ow)
        y = np.zeros((n, c, oh, ow), dtype=np.float32)
        y = conv.im2col_cpu(y, self.p.kh, self.p.kw, self.p.sy, self.p.sx,
                            self.p.ph, self.p.pw)
        rs = list(y.shape)
        y = y.reshape(rs[:2] + [-1] + rs[4:])
        inds = np.asarray(list(itertools.product(np.arange(h), np.arange(w))))
        inds = np.tile(inds, (n * c, 1))
        nc = np.asarray(list(itertools.product(np.arange(n), np.arange(c))))
        pos = self.p.indexes[nc[:, 0], nc[:, 1], :, :].ravel()
        nc = np.repeat(nc, np.prod(self.p.indexes.shape[2:]), axis=0)
        y[nc[:, 0], nc[:, 1], pos, inds[:, 0], inds[:, 1]] = x[0].ravel()
        y = y.reshape(rs)
        y = conv.col2im_cpu(y, self.p.sy, self.p.sx, self.p.ph, self.p.pw,
                            self.outsize[0], self.outsize[1])

        return y,

    def backward(self, x, gy):
        if isinstance(gy[0], cuda.ndarray):
            gcol = conv.im2col_gpu(
                gy[0], self.kh, self.kw, self.sy, self.sx, self.ph, self.pw,
                cover_all=self.cover_all)
        else:
            gcol = conv.im2col_cpu(
                gy[0], self.kh, self.kw, self.sy, self.sx, self.ph, self.pw,
                cover_all=self.cover_all)
        gx = gcol.sum(axis=(2, 3))
        return gx,


def upsampling_2d(x, pooler, outsize):
    """Upsampling using pooling indices.

    This function produces an upsampled image using pooling indices.

    Args:
        x (~chainer.Variable): Input variable.
        pooler (~chainer.functions.Pooling2D): Pooling2D object that is used
            to produce x, the first arg. e.g., An object of MaxPooling2D.
        outsize (pair of ints): Expected output size (height, width).

    Returns:
        ~chainer.Variable: Output variable.

    """
    return Upsampling2D(pooler, outsize)(x)
