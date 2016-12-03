#!/bin/bash

export CHAINER_SEED=2016

init_train () {
    python train.py \
    --seed 2016 --gpu 0 --batchsize 16 \
    --opt MomentumSGD --lr 0.0001 \
    --rotate --fliplr --use_class_weights \
    --show_log_iter 1 \
    --snapshot_epoch 10 \
    --valid_freq 1 \
    --epoch 100 \
    --result_dir results \
    --train_depth 1
}

train () {
    python train.py \
    --seed 2016 --gpu 0 --batchsize 16 \
    --opt Adam --adam_alpha 0.0001 \
    --rotate --fliplr --use_class_weights \
    --show_log_iter 1 \
    --snapshot_epoch 10 \
    --epoch 100 \
    --result_dir results \
    --train_depth $1 \
    --resume $2
}

finetune () {
    python train.py \
    --seed 2016 --gpu 0 --batchsize 16 \
    --opt Adam --adam_alpha 0.00005 \
    --rotate --fliplr --use_class_weights \
    --show_log_iter 1 \
    --snapshot_epoch 10 \
    --epoch 100 \
    --result_dir results \
    --train_depth $1 \
    --finetune \
    --resume $2
}


init_train
train 2 results/encdec1_epoch_100.model
train 3 results/encdec2_epoch_100.model
train 4 results/encdec3_epoch_100.model
finetune 4 results/encdec4_epoch_100.model
