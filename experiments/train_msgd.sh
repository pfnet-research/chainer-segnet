#!/bin/bash

export CHAINER_SEED=2016
export CHAINER_TYPE_CHECK=0

result_dir=results_`date "+%Y-%m-%d_%H%M%S"`
gpu_id=0
batchsize=8
lr=0.0005

init_train () {
    python train.py \
    --seed 2016 --gpus ${gpu_id} --batchsize ${batchsize} \
    --mean data/train_mean.npy \
    --std data/train_std.npy \
    --rotate --fliplr --use_class_weight \
    --opt MomentumSGD --lr ${lr} \
    --snapshot_epoch 10 \
    --valid_freq 10 \
    --epoch 200 \
    --result_dir ${result_dir} \
    --n_encdec 4 \
    --train_depth 1
}

train () {
    python train.py \
    --seed 2016 --gpus ${gpu_id} --batchsize ${batchsize} \
    --mean data/train_mean.npy \
    --std data/train_std.npy \
    --opt MomentumSGD --lr ${lr} \
    --rotate --fliplr --use_class_weight \
    --snapshot_epoch 10 \
    --valid_freq 10 \
    --epoch $3 \
    --result_dir ${result_dir} \
    --n_encdec 4 \
    --train_depth $1 \
    --resume $2
}

finetune () {
    python train.py \
    --seed 2016 --gpus ${gpu_id} --batchsize ${batchsize} \
    --mean data/train_mean.npy \
    --std data/train_std.npy \
    --opt MomentumSGD --lr ${lr} \
    --rotate --fliplr --use_class_weight \
    --snapshot_epoch 10 \
    --valid_freq 10 \
    --epoch $3 \
    --result_dir ${result_dir} \
    --n_encdec 4 \
    --train_depth $1 \
    --finetune \
    --resume $2
}

init_train
train 2 ${result_dir}/encdec1_epoch_200.trainer 400
train 3 ${result_dir}/encdec2_epoch_400.trainer 600
train 4 ${result_dir}/encdec3_epoch_600.trainer 800
finetune 4 ${result_dir}/encdec4_epoch_800.trainer 1000
