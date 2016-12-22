#!/bin/bash

export CHAINER_SEED=2016

result_dir=results_`date "+%Y-%m-%d_%H%M%S"`
gpu_id=1

init_train () {
    python train.py \
    --seed 2016 --gpu ${gpu_id} --batchsize 16 \
    --rotate --fliplr --use_class_weights \
    --opt Adam --adam_alpha 0.0001 \
    --show_log_iter 1 \
    --snapshot_epoch 10 \
    --valid_freq 1 \
    --epoch 100 \
    --result_dir ${result_dir} \
    --n_encdec 4 \
    --train_depth 1
}

train () {
    python train.py \
    --seed 2016 --gpu ${gpu_id} --batchsize 16 \
    --opt Adam --adam_alpha 0.0001 \
    --rotate --fliplr --use_class_weights \
    --show_log_iter 1 \
    --snapshot_epoch 10 \
    --epoch 100 \
    --result_dir ${result_dir} \
    --train_depth $1 \
    --resume $2
}

finetune () {
    python train.py \
    --seed 2016 --gpu ${gpu_id} --batchsize 16 \
    --opt Adam --adam_alpha 0.0001 \
    --rotate --fliplr --use_class_weights \
    --show_log_iter 1 \
    --snapshot_epoch 10 \
    --epoch 100 \
    --result_dir ${result_dir} \
    --train_depth $1 \
    --finetune \
    --resume $2
}


init_train
train 2 ${result_dir}/encdec1_epoch_100.trainer
train 3 ${result_dir}/encdec2_epoch_100.trainer
train 4 ${result_dir}/encdec3_epoch_100.trainer
finetune 4 ${result_dir}/encdec4_epoch_100.model
