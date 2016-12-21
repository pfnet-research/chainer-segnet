#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2016 Shunta Saito

from chainer import cuda
from chainer import optimizers
from chainer import testing
from chainer import Variable
from chainer.computational_graph import build_computational_graph
from models import segnet

import copy
import numpy as np
import os
import re
import six
import subprocess
import unittest

_var_style = {'shape': 'octagon', 'fillcolor': '#E0E0E0', 'style': 'filled'}
_func_style = {'shape': 'record', 'fillcolor': '#6495ED', 'style': 'filled'}

if cuda.available:
    xp = cuda.cupy


@testing.parameterize(*testing.product({
    'n_encdec': [1, 3, 4, 5],
    'n_classes': [2, 3],
    'n_mid': [2, 16, 32],
    'x_shape': [(4, 3, 5, 7), (2, 2, 4, 6)],
}))
class TestSegNet(unittest.TestCase):

    def setUp(self):
        pass

    def get_xt(self):
        x = np.random.uniform(-1, 1, self.x_shape)
        x = Variable(x.astype(np.float32))
        t = np.random.randint(
            0, self.n_classes,
            (self.x_shape[0], self.x_shape[2], self.x_shape[3]))
        t = Variable(t.astype(np.int32))
        return x, t

    def test_remove_link(self):
        opt = optimizers.MomentumSGD(lr=0.01)
        # Update each depth
        for depth in six.moves.range(1, self.n_encdec + 1):
            model = segnet.SegNet(self.n_encdec, self.n_classes,
                                  self.x_shape[1], self.n_mid)
            model = segnet.SegNetLoss(
                model, class_weight=None, train_depth=depth)
            opt.setup(model)

            # Deregister non-target links from opt
            if depth > 1:
                model.predictor.remove_link('conv_cls')
            for d in range(1, self.n_encdec + 1):
                if d != depth:
                    model.predictor.remove_link('encdec{}'.format(d))

            for name, link in model.namedparams():
                if depth > 1:
                    self.assertTrue(
                        'encdec{}'.format(depth) in name)
                else:
                    self.assertTrue(
                        'encdec{}'.format(depth) in name or 'conv_cls' in name)

    def test_backward(self):
        opt = optimizers.MomentumSGD(lr=0.01)
        # Update each depth
        for depth in six.moves.range(1, self.n_encdec + 1):
            model = segnet.SegNet(self.n_encdec, self.n_classes,
                                  self.x_shape[1], self.n_mid)
            model = segnet.SegNetLoss(
                model, class_weight=None, train_depth=depth)
            opt.setup(model)

            # Deregister non-target links from opt
            if depth > 1:
                model.predictor.remove_link('conv_cls')
            for d in range(1, self.n_encdec + 1):
                if d != depth:
                    model.predictor.remove_link('encdec{}'.format(d))

            # Keep the initial values
            prev_params = {
                'conv_cls': copy.deepcopy(model.predictor.conv_cls.W.data)}
            for d in range(1, self.n_encdec + 1):
                name = '/encdec{}/enc/W'.format(d)
                encdec = getattr(model.predictor, 'encdec{}'.format(d))
                prev_params[name] = copy.deepcopy(encdec.enc.W.data)
                self.assertTrue(prev_params[name] is not encdec.enc.W.data)

            # Update the params
            x, t = self.get_xt()
            loss = model(x, t)
            loss.data *= 1E20
            model.cleargrads()
            loss.backward()
            opt.update()

            for d in range(1, self.n_encdec + 1):
                # The weight only in the target layer should be updated
                c = self.assertFalse if d == depth else self.assertTrue
                encdec = getattr(opt.target.predictor, 'encdec{}'.format(d))
                self.assertTrue(hasattr(encdec, 'enc'))
                self.assertTrue(hasattr(encdec.enc, 'W'))
                self.assertTrue('/encdec{}/enc/W'.format(d) in prev_params)
                c(np.array_equal(encdec.enc.W.data,
                                 prev_params['/encdec{}/enc/W'.format(d)]),
                  msg='depth:{} d:{} diff:{}'.format(
                  depth, d, np.sum(encdec.enc.W.data -
                                   prev_params['/encdec{}/enc/W'.format(d)])))
            if depth == 1:
                # The weight in the last layer should be updated
                self.assertFalse(np.allclose(model.predictor.conv_cls.W.data,
                                             prev_params['conv_cls']))

            cg = build_computational_graph(
                [loss],
                variable_style=_var_style,
                function_style=_func_style
            ).dump()

            fn = 'tests/SegNet_bw_depth-{}_{}.dot'.format(self.n_encdec, depth)
            if os.path.exists(fn):
                continue
            with open(fn, 'w') as f:
                f.write(cg)
            subprocess.call(
                'dot -Tpng {} -o {}'.format(
                    fn, fn.replace('.dot', '.png')), shell=True)

            for name, param in model.namedparams():
                encdec_depth = re.search('encdec([0-9]+)', name)
                if encdec_depth:
                    ed = int(encdec_depth.groups()[0])
                    self.assertEqual(ed, depth)

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

            fn = 'tests/SegNet_x_depth-{}_{}.dot'.format(self.n_encdec, depth)
            if os.path.exists(fn):
                continue
            with open(fn, 'w') as f:
                f.write(cg)
            subprocess.call(
                'dot -Tpng {} -o {}'.format(
                    fn, fn.replace('.dot', '.png')), shell=True)

    def test_save_loss_graphs_no_class_weight(self):
        x = np.random.uniform(-1, 1, self.x_shape)
        x = Variable(x.astype(np.float32))
        t = np.random.randint(
            0, 12, (self.x_shape[0], self.x_shape[2], self.x_shape[3]))
        t = Variable(t.astype(np.int32))

        for depth in six.moves.range(1, self.n_encdec + 1):
            model = segnet.SegNet(n_encdec=self.n_encdec, n_classes=12,
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

            fn = 'tests/SegNet_xt_depth-{}_{}.dot'.format(self.n_encdec, depth)
            if os.path.exists(fn):
                continue
            with open(fn, 'w') as f:
                f.write(cg)
            subprocess.call(
                'dot -Tpng {} -o {}'.format(
                    fn, fn.replace('.dot', '.png')), shell=True)
