#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2 as cv
import glob
import numpy as np

fns = glob.glob('data/train/*.png')
n = len(fns)
mean = None
for img_fn in fns:
    img = cv.imread(img_fn)
    if mean is None:
        mean = img.astype(np.float64)
    else:
        mean += img
mean /= n

std = None
for img_fn in fns:
    img = cv.imread(img_fn)
    if std is None:
        std = (img.astype(np.float64) - mean) ** 2
    else:
        std += (img.astype(np.float64) - mean) ** 2
std /= n

np.save('data/train_mean', mean)
np.save('data/train_std', std)
cv.imwrite('data/train_mean.png', mean)
cv.imwrite('data/train_std.png', std)
