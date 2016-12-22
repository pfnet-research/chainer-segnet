# SegNet

SegNet implementation & experiments written in Chainer

This is an unofficial implementation of SegNet. This implementation doesn't use L-BFGS for optimization. This uses Adam with the default settings.

## Requirements

- Python 2.7.12+, 3.5.1+
- Chainer 1.17.0+
- scikit-learn 0.17.1
- NumPy 1.11.0+
- six 1.10.0
- OpenCV 3.1.0
  - `conda install -c https://conda.binstar.org/menpo opencv3`
- Graphviz (To execute tests)
  - `sudo apt-get install -y graphviz`

## Download Dataset

```
bash experiments/download.sh
```

This shell script performs download CamVid dataset from [SegNet-Tutorial](https://github.com/alexgkendall/SegNet-Tutorial) repository owned by the original auther of the SegNet paper.

## Training

```
bash experiments/train.sh
```

### About train.py

To use the given coefficients to weight the softmax loss class-wise, add `--use_class_weights` option to the above command.

Once the first training for the most outer encoder-decoder pair, start the training for the next inner pair from the saved model state.

```
python train.py \
--seed 0 --gpu 1 \
--opt Adam --adam_alpha 0.0001 --batchsize 16 \
--rotate --fliplr --use_class_weights \
--show_log_iter 1 \
--snapshot_epoch 20 \
--epoch 20 \
--result_dir results \
--train_depth 2 \
--resume results/encdec1_epoch_20.model
```

## Prediction

```
python predict.py \
--saved_args results_2016-12-22_130937/args.json \
--snapshot results_2016-12-22_130937/encdec1_epoch_30.trainer \
--out_dir results_2016-12-22_130937/pred_30 \
--gpu 3
```

# Reference

Vijay Badrinarayanan, Alex Kendall and Roberto Cipolla "SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation." arXiv preprint arXiv:1511.00561, 2015\. [PDF](http://arxiv.org/abs/1511.00561)

## Official Implementation with Caffe

- [alexgkendall/SegNet-Tutorial](https://github.com/alexgkendall/SegNet-Tutorial)
- [alexgkendall/caffe-segnet](https://github.com/alexgkendall/caffe-segnet)
