#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2016 Shunta Saito

from chainer import cuda
from chainer import optimizers
from chainer import testing
from chainer import Variable
from chainer.computational_graph import build_computational_graph
from models import segnet

import numpy as np
import os
import six
import subprocess
import unittest

_var_style = {'shape': 'octagon', 'fillcolor': '#E0E0E0', 'style': 'filled'}
_func_style = {'shape': 'record', 'fillcolor': '#6495ED', 'style': 'filled'}

if cuda.available:
    xp = cuda.cupy


@testing.parameterize(*testing.product({
    'n_encdec': [1, 3, 4],
    'n_classes': [1, 4, 5],
    'x_shape': [(4, 3, 5, 7), (1, 2, 4, 6)],
}))
class TestSegNet(unittest.TestCase):

    def setUp(self):
        pass

    def test_save_normal_graphs(self):
        x = np.random.uniform(-1, 1, self.x_shape)
        x = Variable(x.astype(np.float32))

        for depth in six.moves.range(1, self.n_encdec + 1):
            model = segnet.SegNet(
                n_encdec=self.n_encdec, in_channel=self.x_shape[1])
            y = model(x, depth)
            cg = build_computational_graph(
                [y],
                variable_style=_var_style,
                function_style=_func_style
            ).dump()
            for e in range(1, self.n_encdec + 1):
                self.assertTrue('encdec{}'.format(e) in model._children)

            fn = 'tests/SegNet_x_depth-{}.dot'.format(depth)
            if os.path.exists(fn):
                continue
            with open(fn, 'w') as f:
                f.write(cg)
            subprocess.call(
                'dot -Tpng {} -o {}'.format(
                    fn, fn.replace('.dot', '.png')), shell=True)

    def test_save_loss_graphs_no_class_weight(self):
        opt = optimizers.Adam()
        x = np.random.uniform(-1, 1, self.x_shape)
        x = Variable(x.astype(np.float32))
        t = np.random.randint(
            0, 12, (self.x_shape[0], self.x_shape[2], self.x_shape[3]))
        t = Variable(t.astype(np.int32))

        for depth in six.moves.range(1, self.n_encdec + 1):
            model = segnet.SegNet(opt, n_encdec=self.n_encdec, n_classes=12,
                                  in_channel=self.x_shape[1])
            model = segnet.SegNetLoss(
                model, class_weight=None, train_depth=depth)
            y = model(x, t)
            cg = build_computational_graph(
                [y],
                variable_style=_var_style,
                function_style=_func_style
            ).dump()
            for e in range(1, self.n_encdec + 1):
                self.assertTrue(
                    'encdec{}'.format(e) in model.predictor._children)

            fn = 'tests/SegNet_xt_depth-{}.dot'.format(depth)
            if os.path.exists(fn):
                continue
            with open(fn, 'w') as f:
                f.write(cg)
            subprocess.call(
                'dot -Tpng {} -o {}'.format(
                    fn, fn.replace('.dot', '.png')), shell=True)
