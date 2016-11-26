# Copyright (c) 2016 Shunta Saito

from __future__ import division
from __future__ import print_function
from chainer import cuda
from chainer.functions.pooling import pooling_2d
from chainer.utils import conv
from chainer.utils import type_check

import chainer
import cupy
import itertools
import numpy as np
import six


class Upsampling2D(pooling_2d.Pooling2D):

    """Upsampling over a set of 2d planes w/ indices used for max pooling."""

    def __init__(self, indexes, ksize, stride=None, pad=0, outsize=None,
                 cover_all=True):
        super(Upsampling2D, self).__init__(ksize, stride, pad, cover_all)
        self.indexes = indexes
        self.outh, self.outw = (None, None) if outsize is None else outsize

    def check_type_forward(self, in_types):
        n_in = in_types.size()
        type_check.expect(n_in == 1)
        x_type = in_types[0]

        type_check.expect(
            x_type.dtype.kind == 'f',
            x_type.ndim == 4,
            x_type.shape == self.indexes.shape
        )

        if self.outh is not None:
            expected_h = conv.get_conv_outsize(
                self.outh, self.kh, self.sy, self.ph, cover_all=self.cover_all)
            type_check.expect(x_type.shape[2] == expected_h)
        if self.outw is not None:
            expected_w = conv.get_conv_outsize(
                self.outw, self.kw, self.sx, self.pw, cover_all=self.cover_all)
            type_check.expect(x_type.shape[3] == expected_w)

    def forward_cpu(self, x):
        n, c, h, w = x[0].shape
        if self.outh is None:
            self.outh = conv.get_deconv_outsize(
                h, self.kh, self.sy, self.ph, cover_all=self.cover_all)
        if self.outw is None:
            self.outw = conv.get_deconv_outsize(
                w, self.kw, self.sx, self.pw, cover_all=self.cover_all)

        up_y = np.zeros((n, c, self.outh, self.outw), dtype=np.float32)
        up_y = conv.im2col_cpu(up_y, self.kh, self.kw, self.sy,
                               self.sx, self.ph, self.pw)
        for n in six.moves.range(up_y.shape[0]):
            for c in six.moves.range(up_y.shape[1]):
                for ky in six.moves.range(up_y.shape[4]):
                    for kx in six.moves.range(up_y.shape[5]):
                        sy = self.indexes[n, c, ky, kx] / up_y.shape[3]
                        sx = self.indexes[n, c, ky, kx] % up_y.shape[3]
                        up_y[n, c, sy, sx, ky, kx] = x[0][n, c, ky, kx]
        up_y = conv.col2im_cpu(up_y, self.sy, self.sx, self.ph,
                               self.pw, self.outh, self.outw)
        return up_y,

    def forward_gpu(self, x):
        n, c, h, w = x[0].shape
        if self.outh is None:
            self.outh = conv.get_deconv_outsize(
                h, self.kh, self.sy, self.ph, cover_all=self.cover_all)
        if self.outw is None:
            self.outw = conv.get_deconv_outsize(
                w, self.kw, self.sx, self.pw, cover_all=self.cover_all)
        up_y = cupy.zeros((n, c, self.outh, self.outw), dtype=np.float32)
        up_y = conv.im2col_gpu(up_y, self.kh, self.kw, self.sy, self.sx,
                               self.ph, self.pw)
        up_y = up_y.transpose(0, 1, 4, 5, 2, 3)
        yn, yc, ykh, ykw, ysh, ysw = up_y.shape
        indexes = cupy.asarray(self.indexes, dtype=np.int32)
        cupy.ElementwiseKernel(
            'int32 indexes, float32 x, int32 n, int32 c, int32 kh, int32 kw,'
            'int32 sh, int32 sw, int32 xh, int32 xw', 'raw float32 y',
            '''
            int yn = i / c / xh / xw;
            int yc = (i / xh / xw) % c;
            int yky = (i / xw) % xh;
            int ykx = i % xw;
            y[yn * c * kh * kw * sh * sw +
              yc * kh * kw * sh * sw +
              yky * kw * sh * sw +
              ykx * sh * sw +
              indexes] = x;
            ''',
            'upsampling_2d_fwd')(
                indexes, x[0], yn, yc, ykh, ykw, ysh, ysw, h, w, up_y)
        up_y = up_y.transpose(0, 1, 4, 5, 2, 3)
        up_y = conv.col2im_gpu(up_y, self.sy, self.sx, self.ph, self.pw,
                               self.outh, self.outw)
        return up_y,

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


def upsampling_2d(x, indexes, ksize, stride=None, pad=0, outsize=None,
                  cover_all=True):
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
    return Upsampling2D(indexes, ksize, stride, pad, outsize, cover_all)(x)
