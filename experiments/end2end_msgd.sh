#!/bin/bash

export CHAINER_SEED=2016

result_dir=results_`date "+%Y-%m-%d_%H%M%S"`
gpu_id=0,1,2,3
n_encdec=4
batchsize=64

python train.py \
--seed 2016 --gpus ${gpu_id} --batchsize ${batchsize} \
--opt MomentumSGD --lr 0.01 \
--rotate --fliplr --use_class_weight \
--snapshot_epoch 10 \
--epoch 200 \
--result_dir ${result_dir} \
--n_encdec ${n_encdec} \
--train_depth ${n_encdec} \
--finetune

python train.py \
--seed 2016 --gpus ${gpu_id} --batchsize ${batchsize} \
--opt MomentumSGD --lr 0.001 \
--rotate --fliplr --use_class_weight \
--snapshot_epoch 10 \
--epoch 400 \
--result_dir ${result_dir} \
--n_encdec ${n_encdec} \
--train_depth ${n_encdec} \
--finetune \
--resume ${result_dir}/encdec4_finetune_epoch_200.trainer

python train.py \
--seed 2016 --gpus ${gpu_id} --batchsize ${batchsize} \
--opt MomentumSGD --lr 0.0001 \
--rotate --fliplr --use_class_weight \
--snapshot_epoch 10 \
--epoch 600 \
--result_dir ${result_dir} \
--n_encdec ${n_encdec} \
--train_depth ${n_encdec} \
--finetune \
--resume ${result_dir}/encdec4_finetune_epoch_400.trainer

python train.py \
--seed 2016 --gpus ${gpu_id} --batchsize ${batchsize} \
--opt MomentumSGD --lr 0.00001 \
--rotate --fliplr --use_class_weight \
--snapshot_epoch 10 \
--epoch 1000 \
--result_dir ${result_dir} \
--n_encdec ${n_encdec} \
--train_depth ${n_encdec} \
--finetune \
--resume ${result_dir}/encdec4_finetune_epoch_600.trainer
