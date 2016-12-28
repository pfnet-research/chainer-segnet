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
std = np.sqrt(std)

np.save('data/train_mean', mean)
np.save('data/train_std', std)
cv.imwrite('data/train_mean.png', mean)
cv.imwrite('data/train_std.png', std)

class_ids = set()
fns = glob.glob('data/trainannot/*.png')
n = len(fns)
classes = [0.0 for _ in range(12)]
for lbl_fn in fns:
    lbl = cv.imread(lbl_fn, cv.IMREAD_GRAYSCALE)
    for lb_i in np.unique(lbl):
        class_ids.add(lb_i)
    for l in range(12):
        classes[l] += np.sum(lbl == l)
class_freq = np.array(classes)
class_freq /= np.sum(class_freq)
print('Existing class IDs: {}'.format(class_ids))

with open('data/train_freq.csv', 'w') as fp:
    for i, freq in enumerate(class_freq):
        print(1.0 / freq, end=',' if i < len(class_freq) - 1 else '', file=fp)
print('0.2595,0.1826,4.5640,0.1417,0.9051,0.3826,9.6446,1.8418,'
      '0.6823,6.2478,7.3614,0.0', file=open('data/train_origcw.csv', 'w'))
