# Copyright (c) 2016 Shunta Saito

from __future__ import division
from __future__ import print_function
from chainer import cuda
from chainer.utils import conv
from chainer.utils import type_check

import chainer
import cupy
import itertools
import numpy as np
import six


class Upsampling2D(chainer.Function):

    """Upsampling over a set of 2d planes w/ indices used for max pooling."""

    def __init__(self, pooler, outsize):
        self.p = pooler
        self.outsize = outsize

    def forward_cpu(self, x):
        n, c, h, w = x[0].shape
        oh, ow = self.outsize
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

    def forward_gpu(self, x):
        xp = cupy
        n, c, h, w = x[0].shape
        oh, ow = self.outsize
        y = xp.zeros((n, c, oh, ow), dtype=xp.float32)
        y = conv.im2col_gpu(y, self.p.kh, self.p.kw, self.p.sy, self.p.sx,
                            self.p.ph, self.p.pw)
        y = y.transpose(0, 1, 4, 5, 2, 3)
        yn, yc, ykh, ykw, ysh, ysw = y.shape
        indexes = xp.asarray(self.p.indexes, dtype=xp.int32)

        cupy.ElementwiseKernel(
            'int32 indexes, float32 x, int32 yn, int32 yc, int32 ykh,'
            'int32 ykw, int32 ysh, int32 ysw',
            'raw float32 y',
            '''
            int yn_i = i / yc / ykh / ykw;
            int yc_i = (i / ykh / ykw) % yc;
            int yy_i = (i / ykw) % ykh;
            int yx_i = i % ykw;
            y[yn_i * yc * ykh * ykw * ysh * ysw +
              yc_i * ykh * ykw * ysh * ysw +
              yy_i * ykw * ysh * ysw +
              yx_i * ysh * ysw +
              indexes] = x;
            ''',
            'upsampling_2d_fwd')(indexes, x[0], yn, yc, ykh, ykw, ysh, ysw, y)
        y = y.transpose(0, 1, 4, 5, 2, 3)
        y = conv.col2im_gpu(y, self.p.sy, self.p.sx, self.p.ph, self.p.pw,
                            self.outsize[0], self.outsize[1])
        return y,

    def backward_cpu(self, x, gy):
        gcol = conv.im2col_cpu(
            gy[0], self.p.kh, self.p.kw, self.p.sy,
            self.p.sx, self.p.ph, self.p.pw)

        gcol = gcol.transpose(0, 1, 4, 5, 2, 3)
        n, c, ky, kx, sy, sx = gcol.shape
        gcol = gcol.reshape((n, c, ky, kx, sy * sx))
        gx = np.empty((n, c, ky, kx), dtype=x[0].dtype)
        for n in six.moves.range(gcol.shape[0]):
            for c in six.moves.range(gcol.shape[1]):
                for ky in six.moves.range(gcol.shape[2]):
                    for kx in six.moves.range(gcol.shape[3]):
                        gx[n, c, ky, kx] = \
                            gcol[n, c, ky, kx][self.p.indexes[n, c, ky, kx]]

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
