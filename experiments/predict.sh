#!/bin/bash

result_dir=results
n_encdec=4
epoch=1000
gpu=0

python predict.py \
--saved_args ${result_dir}/args_${n_encdec}.json \
--snapshot ${result_dir}/encdec${n_encdec}_epoch_${epoch}.trainer \
--out_dir ${result_dir}/pred_encdec${n_encdec}_epoch${epoch} \
--gpu ${gpu}
