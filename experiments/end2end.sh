#!/bin/bash

export CHAINER_SEED=2016

result_dir=results_`date "+%Y-%m-%d_%H%M%S"`
gpu_id=0

end_to_end () {
    python train.py \
    --seed 2016 --gpu ${gpu_id} --batchsize 16 \
    --opt Adam --adam_alpha 0.0001 \
    --rotate --fliplr --use_class_weight \
    --snapshot_epoch 10 \
    --epoch $2 \
    --result_dir ${result_dir} \
    --n_encdec $1 \
    --train_depth $1 \
    --finetune
}

end_to_end 4 1000
