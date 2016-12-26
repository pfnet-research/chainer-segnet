#!/usr/bin/env python
# -*- coding: utf-8 -*-

import glob
import json
import os
import re
import subprocess

for result_dir in glob.glob('results*'):
    if not os.path.isdir(result_dir):
        continue
    args_fn = glob.glob('{}/arg*.json'.format(result_dir))[0]
    args = json.load(open(args_fn))

    pmfn = sorted([(int(re.search('epoch_([0-9]+)', fn).groups()[0]), fn)
                   for fn in glob.glob('{}/*.trainer'.format(result_dir))])
    pmfn = pmfn[-1][1]
    pred_dname = result_dir + \
        '/pred_' + os.path.splitext(os.path.basename(pmfn))[0]

    cmd = ['python', 'predict.py',
           '--train_depth', '4',
           '--saved_args', '{}/args_4.json'.format(result_dir),
           '--snapshot', pmfn,
           '--out_dir', pred_dname]
    if args['mean'] is not None:
        cmd += ['--mean', args['mean']]
        cmd += ['--std', args['std']]
    subprocess.call(cmd)
