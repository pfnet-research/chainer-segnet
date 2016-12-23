#!/bin/bash

snapshot_epoch=2 epoch=2 bash experiments/train.sh > tests/log_train
snapshot_epoch=2 epoch=2 bash experiments/train_nostd.sh > tests/log_train_nostd
snapshot_epoch=2 epoch=2 bash experiments/train_nostd_nocw.sh > tests/log_train_nostd_nocw
snapshot_epoch=2 epoch=2 bash experiments/train_noda.sh > tests/log_train_noda
snapshot_epoch=2 epoch=2 bash experiments/train_nocw.sh > tests/log_train_nocw
snapshot_epoch=2 epoch=2 bash experiments/train_msgd.sh > tests/log_train_msgd
snapshot_epoch=2 epoch=2 bash experiments/train_end2end.sh > tests/log_train_end2end
snapshot_epoch=2 epoch=2 bash experiments/train_end2end_noda.sh > tests/log_train_end2end_noda
snapshot_epoch=2 epoch=2 bash experiments/train_end2end_msgd.sh > tests/log_train_end2end_msgd
