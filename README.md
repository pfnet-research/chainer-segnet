# SegNet
SegNet implementation &amp; experiments written in Chainer

This is an unofficial implementation of SegNet. This implementation doesn't use L-BFGS for optimization. This uses Adam with the default settings.

## Requirements

- Python 2.7.12 (Python 3.5.1+ somehow stacks at multiprocessing)
- Chainer 1.17.0+
- scikit-learn 0.17.1
- NumPy 1.11.0+
- OpenCV 3.1.0
    - `conda install -c https://conda.binstar.org/menpo opencv3`

## Dataset Download

```
wget https://dl.dropboxusercontent.com/u/2498135/segnet/CamVid.tar.gz
tar zxvf CamVid.tar.gz; rm -rf CamVid.tar.gz; mv CamVid data
```

## Training

```
python train.py --gpus 0 --batchsize 16 --rotate --fliplr
```

To use the given coefficients to weight the softmax loss class-wise, add `--use_class_weights` option to the above command.

# Citation

Vijay Badrinarayanan, Alex Kendall and Roberto Cipolla "SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation." arXiv preprint arXiv:1511.00561, 2015. [PDF](http://arxiv.org/abs/1511.00561)

## Official Implementation with Caffe

- [alexgkendall/SegNet-Tutorial](https://github.com/alexgkendall/SegNet-Tutorial)
- [alexgkendall/caffe-segnet](https://github.com/alexgkendall/caffe-segnet)
