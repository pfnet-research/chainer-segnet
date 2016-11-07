#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2016 Shunta Saito

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from chainer import Variable
from models import upsampling_2d

import chainer.functions as F
import numpy as np
import unittest


class TestUpsampling2D(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.x = Variable(np.asarray([[[
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
            [2., 4., 3., 4., 4., 4.]]]]).astype(np.float32))
        cls.p = F.MaxPooling2D(2, 2)
        cls.y = cls.p(cls.x)
        cls.outsize = (6, 6)
        cls.upsampled_y = np.asarray([[[
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

    def test_forward_cpu(self):
        y = upsampling_2d.upsampling_2d(self.y, self.p, self.outsize)
        self.assertEqual(y.data.shape, self.upsampled_y.shape)
        self.assertEqual(y.data.dtype, np.float32)
        self.assertEqual(self.upsampled_y.dtype, np.float32)
        self.assertTrue(np.all(y.data == self.upsampled_y))
