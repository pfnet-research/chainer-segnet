#!/bin/bash

python predict.py \
--model_file models/segnet.py \
--model_name SegNet \
--n_classes 12 \
--n_encdec 4 \
--snapshot results/encdec4_epoch_20.model \
--img_dir data/val \
--out_dir results/pred_ed4ep20 \
--gpu 0
