#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2016 Shunta Saito

from chainer import optimizers
from chainer import Variable
from chainer.computational_graph import build_computational_graph
from models import segnet

import cupy as xp
import numpy as np

_var_style = {'shape': 'octagon', 'fillcolor': '#E0E0E0', 'style': 'filled'}
_func_style = {'shape': 'record', 'fillcolor': '#6495ED', 'style': 'filled'}


if __name__ == '__main__':
    opt = optimizers.Adam()

    model = segnet.SegNet(opt)
    model = segnet.SegNetLoss(model, [1 for _ in range(12)])
    model.to_gpu()

    for depth in range(1, 5):
        print('depth:', depth)
        x = np.random.uniform(-1, 1, (4, 3, 360, 480))
        x = Variable(xp.asarray(x.astype(np.float32)))
        t = np.random.randint(0, 12, size=(4, 360, 480))
        t = Variable(xp.asarray(t.astype(np.int32)))
        loss, optimizers = model(x, t, depth)
        print(loss.debug_print())

        cg = build_computational_graph(
            [loss],
            variable_style=_var_style,
            function_style=_func_style
        ).dump()
        with open('SegNet-depth{}.dot'.format(depth), 'w') as f:
            f.write(cg)
