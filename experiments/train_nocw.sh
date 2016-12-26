#!/bin/bash

seed=2016
gpu_id=0
batchsize=16
opt=Adam
lr=0.001
adam_alpha=0.0001
n_encdec=4
result_dir=results_NoCW_opt-${opt}_lr-${lr}_alpha-${adam_alpha}_`date "+%Y-%m-%d_%H%M%S"`

if [ -z ${snapshot_epoch} ]; then
    snapshot_epoch=10
fi
if [ -z ${epoch} ]; then
    epoch=300
fi

init_train () {
    CHAINER_SEED=${seed} CHAINER_TYPE_CHECK=0 python train.py \
    --seed 2016 \
    --gpus ${gpu_id} \
    --batchsize ${batchsize} \
    --opt ${opt} \
    --lr ${lr} \
    --adam_alpha ${adam_alpha} \
    --snapshot_epoch ${snapshot_epoch} \
    --valid_freq ${snapshot_epoch} \
    --result_dir ${result_dir} \
    --n_encdec ${n_encdec} \
    --mean data/train_mean.npy \
    --std data/train_std.npy \
    --rotate \
    --fliplr \
    --epoch $1 \
    --train_depth 1
}

train () {
    CHAINER_SEED=${seed} CHAINER_TYPE_CHECK=0 python train.py \
    --seed 2016 \
    --gpus ${gpu_id} \
    --batchsize ${batchsize} \
    --opt ${opt} \
    --lr ${lr} \
    --adam_alpha ${adam_alpha} \
    --snapshot_epoch ${snapshot_epoch} \
    --valid_freq ${snapshot_epoch} \
    --result_dir ${result_dir} \
    --n_encdec ${n_encdec} \
    --mean data/train_mean.npy \
    --std data/train_std.npy \
    --rotate \
    --fliplr \
    --train_depth $1 \
    --resume $2 \
    --epoch $3
}

finetune () {
    CHAINER_SEED=${seed} CHAINER_TYPE_CHECK=0 python train.py \
    --seed 2016 \
    --gpus ${gpu_id} \
    --batchsize ${batchsize} \
    --opt ${opt} \
    --lr ${lr} \
    --adam_alpha ${adam_alpha} \
    --snapshot_epoch ${snapshot_epoch} \
    --valid_freq ${snapshot_epoch} \
    --result_dir ${result_dir} \
    --n_encdec ${n_encdec} \
    --mean data/train_mean.npy \
    --std data/train_std.npy \
    --rotate \
    --fliplr \
    --train_depth $1 \
    --resume $2 \
    --epoch $3 \
    --finetune
}

init_train ${epoch}
train 2 ${result_dir}/encdec1_epoch_${epoch}.trainer `expr ${epoch} \* 2`
train 3 ${result_dir}/encdec2_epoch_`expr ${epoch} \* 2`.trainer `expr ${epoch} \* 3`
train 4 ${result_dir}/encdec3_epoch_`expr ${epoch} \* 3`.trainer `expr ${epoch} \* 4`
finetune 4 ${result_dir}/encdec4_epoch_`expr ${epoch} \* 4`.trainer `expr ${epoch} \* 5`
