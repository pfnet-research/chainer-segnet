#!/bin/bash

seed=2016
gpu_id=0
batchsize=16
opt=MomentumSGD
lr=0.01
adam_alpha=0.0001
n_encdec=4
result_dir=results_end2end_opt-${opt}_lr-${lr}_alpha-${adam_alpha}_`date "+%Y-%m-%d_%H%M%S"`

if [ -z ${snapshot_epoch} ]; then
    snapshot_epoch=10
fi
if [ -z ${epoch} ]; then
    epoch=200
fi

init_train () {
    CHAINER_SEED=${seed} CHAINER_TYPE_CHECK=0 python train.py \
    --seed 2016 \
    --gpus ${gpu_id} \
    --batchsize ${batchsize} \
    --opt ${opt} \
    --adam_alpha ${adam_alpha} \
    --snapshot_epoch ${snapshot_epoch} \
    --valid_freq ${snapshot_epoch} \
    --result_dir ${result_dir} \
    --n_encdec ${n_encdec} \
    --train_depth ${n_encdec} \
    --mean data/train_mean.npy \
    --std data/train_std.npy \
    --finetune \
    --use_class_weight \
    --rotate \
    --fliplr \
    --lr $1 \
    --epoch $2
}

train () {
    CHAINER_SEED=${seed} CHAINER_TYPE_CHECK=0 python train.py \
    --seed 2016 \
    --gpus ${gpu_id} \
    --batchsize ${batchsize} \
    --opt ${opt} \
    --adam_alpha ${adam_alpha} \
    --snapshot_epoch ${snapshot_epoch} \
    --valid_freq ${snapshot_epoch} \
    --result_dir ${result_dir} \
    --n_encdec ${n_encdec} \
    --train_depth ${n_encdec} \
    --mean data/train_mean.npy \
    --std data/train_std.npy \
    --finetune \
    --use_class_weight \
    --rotate \
    --fliplr \
    --lr $1 \
    --epoch $2 \
    --resume $3
}

init_train 0.01 ${epoch}
train 0.001 `expr ${epoch} \* 2` ${result_dir}/encdec4_finetune_epoch_${epoch}.trainer
train 0.0005 `expr ${epoch} \* 3` ${result_dir}/encdec4_finetune_epoch_`expr ${epoch} \* 2`.trainer
train 0.0001 `expr ${epoch} \* 4` ${result_dir}/encdec4_finetune_epoch_`expr ${epoch} \* 3`.trainer
train 0.00005 `expr ${epoch} \* 5` ${result_dir}/encdec4_finetune_epoch_`expr ${epoch} \* 4`.trainer
