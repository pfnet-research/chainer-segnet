#!/bin/bash

export CHAINER_SEED=0

init_train () {
    python train.py \
    --seed 0 --gpu 0 \
    --opt Adam --adam_alpha 0.0001 --batchsize 16 \
    --rotate --fliplr --use_class_weights \
    --show_log_iter 1 \
    --snapshot_epoch 20 \
    --epoch 20 \
    --result_dir results \
    --train_depth 1
}

train () {
    python train.py \
    --seed 0 --gpu 0 \
    --opt Adam --adam_alpha 0.0001 --batchsize 16 \
    --rotate --fliplr --use_class_weights \
    --show_log_iter 1 \
    --snapshot_epoch 20 \
    --epoch 20 \
    --result_dir results \
    --train_depth $1 \
    --resume $2
}


init_train
train 2 results/encdec1_epoch_20.model
train 3 results/encdec2_epoch_20.model
train 4 results/encdec3_epoch_20.model
