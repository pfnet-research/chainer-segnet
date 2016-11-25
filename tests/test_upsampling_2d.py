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
from chainer import Variable
from chainer.testing import attr
from chainer.testing import condition
from models import upsampling_2d

import chainer.functions as F
import cupy
import numpy as np
import unittest


class TestUpsampling2D(unittest.TestCase):

    def setUp(self):
        self.x = np.asarray([[[
            [4., 0., 3., 3., 3., 1.],
            [3., 2., 4., 0., 0., 4.],
            [2., 1., 0., 1., 1., 0.],
            [1., 4., 3., 0., 3., 0.],
            [2., 3., 0., 1., 3., 3.],
            [3., 0., 1., 1., 1., 0.]
        ], [
            [2., 4., 3., 3., 2., 4.],
            [2., 0., 0., 4., 0., 4.],
            [1., 4., 1., 2., 2., 0.],
            [1., 1., 1., 1., 3., 3.],
            [2., 3., 0., 3., 4., 1.],
            [2., 4., 3., 4., 4., 4.]]]]).astype(np.float32)
        self.p = F.MaxPooling2D(2, 2)
        self.y = self.p(self.x)
        self.outsize = (6, 6)
        self.upsampled_y = np.asarray([[[
            [4., 0., 0., 0., 0., 0.],
            [0., 0., 4., 0., 0., 4.],
            [0., 0., 0., 0., 0., 0.],
            [0., 4., 3., 0., 3., 0.],
            [0., 3., 0., 1., 3., 0.],
            [0., 0., 0., 0., 0., 0.]],
            [[0., 4., 0., 0., 0., 4.],
             [0., 0., 0., 4., 0., 0.],
             [0., 4., 0., 2., 0., 0.],
             [0., 0., 0., 0., 3., 0.],
             [0., 0., 0., 0., 4., 0.],
             [0., 4., 0., 4., 0., 0.]]]]).astype(np.float32)
        self.gy = np.random.uniform(
            -1, 1, self.upsampled_y.shape).astype(np.float32)

    def check_forward(self, y):
        xp = cuda.get_array_module(y.data)
        y = upsampling_2d.upsampling_2d(y, self.p, self.outsize)
        self.assertEqual(y.data.shape, self.upsampled_y.shape)
        self.assertEqual(y.data.dtype, xp.float32)
        self.assertEqual(self.upsampled_y.dtype, xp.float32)
        if isinstance(y.data, cupy.ndarray):
            self.upsampled_y = xp.asarray(self.upsampled_y, dtype=xp.float32)
        self.assertTrue(xp.all(y.data == self.upsampled_y))

    @condition.retry(3)
    def test_forward_cpu(self):
        self.y.to_cpu()
        self.check_forward(self.y)

    @attr.gpu
    @condition.retry(3)
    def test_forward_gpu(self):
        self.y.to_gpu()
        self.check_forward(self.y)

    def check_backward(self, x_data, y_grad):
        func = upsampling_2d.Upsampling2D(self.p, self.outsize)
        gradient_check.check_backward(func, x_data, y_grad)

    @condition.retry(3)
    def test_backward_cpu(self):
        self.check_backward(self.y.data, self.gy)

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu(self):
        self.check_backward(cuda.to_gpu(self.y.data), cuda.to_gpu(self.gy))
