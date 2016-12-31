#!/usr/bin/env python
# -*- coding: utf-8 -*-

import glob
import os
import subprocess

for result_dir in glob.glob('results*/pred*'):
    if not os.path.isdir(result_dir):
        continue
    print(result_dir)
    cmd = ['python', 'evaluate.py',
           '--result_dir', result_dir]
    subprocess.call(cmd)
    print('-' * 20)
