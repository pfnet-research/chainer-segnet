# SegNet
SegNet implementation &amp; experiments written in Chainer

This is an unofficial implementation of SegNet. This implementation doesn't use L-BFGS for optimization. This uses Adam with the default settings.

## Requirements

- Python 2.7.12+, 3.5.1+
- Chainer 1.17.0+
- scikit-learn 0.17.1
- NumPy 1.11.0+
- six 1.10.0
- OpenCV 3.1.0
    - `conda install -c https://conda.binstar.org/menpo opencv3`

## Dataset Download

```
wget https://dl.dropboxusercontent.com/u/2498135/segnet/CamVid.tar.gz
tar zxvf CamVid.tar.gz; rm -rf CamVid.tar.gz; mv CamVid data
```

## Training

```
python train.py \
--opt Adam \
--gpu 0 \
--batchsize 16 \
--rotate \
--fliplr \
--use_class_weights \
--show_log_iter 1 \
--snapshot_epoch 10 \
--epoch 20 \
--train_depth 1
```

To use the given coefficients to weight the softmax loss class-wise, add `--use_class_weights` option to the above command.

Once the first training for the most outer encoder-decoder pair, start the training for the next inner pair from the saved model state.

```
python train.py \
--opt Adam \
--gpu 0 \
--batchsize 16 \
--rotate \
--fliplr \
--use_class_weights \
--show_log_iter 1 \
--snapshot_epoch 10 \
--train_depth 2 \
--resume results/****/EncDec1_epoch20
```


# Citation

Vijay Badrinarayanan, Alex Kendall and Roberto Cipolla "SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation." arXiv preprint arXiv:1511.00561, 2015. [PDF](http://arxiv.org/abs/1511.00561)

## Official Implementation with Caffe

- [alexgkendall/SegNet-Tutorial](https://github.com/alexgkendall/SegNet-Tutorial)
- [alexgkendall/caffe-segnet](https://github.com/alexgkendall/caffe-segnet)
