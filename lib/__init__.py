#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2016 Shunta Saito

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from lib import camvid
from lib import cmd_options
from lib import train_utils

CamVid = camvid.CamVid

get_args = cmd_options.get_args

create_result_dir = train_utils.create_result_dir
create_logger = train_utils.create_logger
get_model = train_utils.get_model
get_optimizer = train_utils.get_optimizer
