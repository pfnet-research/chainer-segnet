#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2016 Shunta Saito

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from chainer import functions as F
from chainer import cuda
from chainer import gradient_check
from chainer import testing
from chainer import Variable
from chainer.testing import attr
from chainer.testing import condition
from chainer.utils import conv
from models import upsampling_2d

import chainer.functions as F
import cupy
import numpy as np
import six
import unittest


@testing.parameterize(
    {'in_shape': (2, 3, 6, 6)},
    {'in_shape': (2, 4, 8, 8)},
    {'in_shape': (2, 4, 10, 10)},
    {'in_shape': (2, 4, 10, 12)},
    # {'in_shape': (2, 3, 7, 7)},
    # {'in_shape': (2, 4, 5, 6)},
)
class TestUpsampling2D(unittest.TestCase):

    def setUp(self):
        self.x = np.random.uniform(-1, 1, self.in_shape).astype('f')
        self.p = F.MaxPooling2D(2, 2)
        self.pooled_y = self.p(self.x)
        self.gy = np.random.uniform(
            -1, 1, self.in_shape).astype(np.float32)

    def check_forward(self, y):
        y = upsampling_2d.upsampling_2d(
            self.pooled_y, self.p.indexes, ksize=(self.p.kh, self.p.kw),
            stride=(self.p.sy, self.p.sx), pad=(self.p.ph, self.p.pw),
            outsize=self.in_shape[2:], cover_all=self.p.cover_all)
        if isinstance(y.data, np.ndarray):
            y = conv.im2col_cpu(y.data, self.p.kh, self.p.kw,
                                self.p.sy, self.p.sx, self.p.ph, self.p.pw)
        else:
            y = conv.im2col_gpu(y.data, self.p.kh, self.p.kw,
                                self.p.sy, self.p.sx, self.p.ph, self.p.pw)
        # print('y:', y)
        # print('pooled_y:', self.pooled_y.data)
        for n in six.moves.range(y.shape[0]):
            for c in six.moves.range(y.shape[1]):
                for ky in six.moves.range(y.shape[4]):
                    for kx in six.moves.range(y.shape[5]):
                        for sy in six.moves.range(y.shape[2]):
                            for sx in six.moves.range(y.shape[3]):
                                up_y = y[n, c, sy, sx, ky, kx]
                                if sy * y.shape[3] + sx == \
                                        self.p.indexes[n, c, ky, kx]:
                                    in_y = self.pooled_y.data[n, c, ky, kx]
                                    testing.assert_allclose(in_y, up_y)
                                else:
                                    testing.assert_allclose(up_y, 0)

    @condition.retry(3)
    def test_forward_cpu(self):
        self.pooled_y.to_cpu()
        self.check_forward(self.pooled_y)

    @attr.gpu
    @condition.retry(3)
    def test_forward_gpu(self):
        self.pooled_y.to_gpu()
        self.check_forward(self.pooled_y)
    #
    # def check_backward(self, x_data, y_grad):
    #     func = upsampling_2d.Upsampling2D(self.p, self.outsize)
    #     gradient_check.check_backward(func, x_data, y_grad)
    #
    # @condition.retry(3)
    # def test_backward_cpu(self):
    #     self.check_backward(self.y.data, self.gy)
    #
    # @attr.gpu
    # @condition.retry(3)
    # def test_backward_gpu(self):
    #     self.check_backward(cuda.to_gpu(self.y.data), cuda.to_gpu(self.gy))
