#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2016 Shunta Saito

from chainer import Variable
from models import segnet

import cupy as xp
import numpy as np

if __name__ == '__main__':
    model = segnet.SegNet(12)
    model.to_gpu()
    x = np.random.uniform(-1, 1, (4, 3, 360, 480))
    x = xp.asarray(x.astype(np.float32))
    x = Variable(x)
    y = model(x)
    print(y.debug_print())
