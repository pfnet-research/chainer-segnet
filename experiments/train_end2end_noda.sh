#!/bin/bash

seed=2016
gpu_id=0
batchsize=16
opt=Adam
lr=0.001
adam_alpha=0.0001
n_encdec=4
result_dir=results_end2end_noda_opt-${opt}_lr-${lr}_alpha-${adam_alpha}_`date "+%Y-%m-%d_%H%M%S"`

if [ -z ${snapshot_epoch} ]; then
    snapshot_epoch=10
fi
if [ -z ${epoch} ]; then
    epoch=1000
fi

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
--train_depth ${n_encdec} \
--epoch ${epoch} \
--mean data/train_mean.npy \
--std data/train_std.npy \
--use_class_weight \
--shift_jitter 0 \
--scale_jitter 0.0 \
--finetune
