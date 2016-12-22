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

### About train.sh

To use the preliminarily calculated coefficients to weight the softmax cross entropy in class-wise manner, add `--use_class_weights` option to the above command.

What the shell script `train.sh` do is a simple sequential training process. Once the first training for a most outer encoder-decoder pair, start the training for the next inner pair from the saved model state of the previous training process.

If you would like to change the training settings and hyper parameters, please see the output of `python train.py --help` to check the argments it can take.

## Prediction

Use `predict.py` to create prediction results on the test dataset.

```
python predict.py \
--saved_args results_2016-04-03_152217/args.json \
--snapshot results_2016-04-03_152217/encdec1_epoch_30.trainer \
--out_dir results_2016-04-03_152217/pred_30 \
--gpu 0
```

# Reference

> Vijay Badrinarayanan, Alex Kendall and Roberto Cipolla "SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation." arXiv preprint arXiv:1511.00561, 2015\. [PDF](http://arxiv.org/abs/1511.00561)

## Official Implementation with Caffe

- [alexgkendall/SegNet-Tutorial](https://github.com/alexgkendall/SegNet-Tutorial)
- [alexgkendall/caffe-segnet](https://github.com/alexgkendall/caffe-segnet)
