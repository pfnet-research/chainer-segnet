#!/bin/bash

export CHAINER_SEED=0

python train.py \
--seed 0 \
--opt Adam \
--gpu 0 \
--batchsize 16 \
--rotate \
--fliplr \
--use_class_weights \
--show_log_iter 1 \
--snapshot_epoch 10 \
--epoch 20 \
--result_dir results \
--train_depth 1

python train.py \
--seed 0 \
--opt Adam \
--gpu 0 \
--batchsize 16 \
--rotate \
--fliplr \
--use_class_weights \
--show_log_iter 1 \
--snapshot_epoch 10 \
--epoch 40 \
--result_dir results \
--train_depth 1 \
--resume results/EncDec1_epoch20
