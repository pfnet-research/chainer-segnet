#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2016 Shunta Saito

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from chainer import cuda
from chainer import functions
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition
from models import weighted_softmax_cross_entropy

import chainer
import mock
import numpy
import six
import unittest


@testing.parameterize(*(testing.product({
    'shape': [(2, 11, 8, 12)],
    'cache_score': [True, False],
    'normalize': [True, False],
    'dtype': [numpy.float32],
    'ignore_index': [None, (0, 1, 0)],
})))
class TestWeightedSoftmaxCrossEntropy(unittest.TestCase):

    def setUp(self):
        self.x = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
        out_shape = (self.shape[0],) + self.shape[2:]
        self.t = numpy.random.randint(0, 3, out_shape).astype(numpy.int32)
        self.w = numpy.array(
            [0.2595, 0.1826, 4.5640, 0.1417, 0.9051, 0.3826, 9.6446, 1.8418,
             0.6823, 6.2478, 7.3614], dtype=numpy.float32)
        if (self.ignore_index is not None and
                len(self.ignore_index) <= self.t.ndim):
            self.t[self.ignore_index] = -1
        self.check_forward_options = {}
        self.check_backward_options = {
            'dtype': numpy.float64, 'atol': 1e-1, 'rtol': 1e-4}

    def check_forward(self, x_data, t_data, w_data, use_cudnn=True):
        x = chainer.Variable(x_data)
        t = chainer.Variable(t_data)
        w = chainer.Variable(w_data)
        loss = weighted_softmax_cross_entropy.weighted_softmax_cross_entropy(
            x, t, w, use_cudnn=use_cudnn, normalize=self.normalize,
            cache_score=self.cache_score)
        self.assertEqual(loss.data.shape, ())
        self.assertEqual(loss.data.dtype, self.dtype)
        self.assertEqual(hasattr(loss.creator, 'y'), self.cache_score)
        loss_value = float(cuda.to_cpu(loss.data))

        # Compute expected value
        loss_expect = 0.0
        count = 0
        x = numpy.rollaxis(self.x, 1, self.x.ndim).reshape(
            (self.t.size, self.x.shape[1]))
        t = self.t.ravel()
        for xi, ti in six.moves.zip(x, t):
            if ti == -1:
                continue
            log_z = numpy.ufunc.reduce(numpy.logaddexp, xi)
            loss_expect -= (xi - log_z)[ti] * w_data[ti]
            count += 1

        if self.normalize:
            if count == 0:
                loss_expect = 0.0
            else:
                loss_expect /= count
        else:
            loss_expect /= len(t_data)

        testing.assert_allclose(
            loss_expect, loss_value, **self.check_forward_options)

    @condition.retry(3)
    def test_forward_cpu(self):
        self.check_forward(self.x, self.t, self.w)

    @attr.gpu
    @condition.retry(3)
    def test_forward_gpu(self):
        self.check_forward(
            cuda.to_gpu(self.x), cuda.to_gpu(self.t), cuda.to_gpu(self.w))

    @attr.gpu
    @condition.retry(3)
    def test_forward_gpu_no_cudnn(self):
        self.check_forward(
            cuda.to_gpu(self.x), cuda.to_gpu(self.t), cuda.to_gpu(self.w),
            False)

    def check_backward(self, x_data, t_data, w_data, use_cudnn=True):
        gradient_check.check_backward(
            weighted_softmax_cross_entropy.WeightedSoftmaxCrossEntropy(
                use_cudnn=use_cudnn, cache_score=self.cache_score),
            (x_data, t_data, w_data), None, eps=0.001,
            no_grads=[False, True, True], **self.check_backward_options)

    @condition.retry(3)
    def test_backward_cpu(self):
        self.check_backward(self.x, self.t, self.w)

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu(self):
        self.check_backward(
            cuda.to_gpu(self.x), cuda.to_gpu(self.t), cuda.to_gpu(self.w))

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu_no_cudnn(self):
        self.check_backward(
            cuda.to_gpu(self.x), cuda.to_gpu(self.t), cuda.to_gpu(self.w),
            False)

testing.run_module(__name__, __file__)
