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

## Calculate dataset mean

```
python lib/calc_mean.py
```

It produces `train_mean.npy` and `train_std.npy` to normalize inputs during training and also `train_freq.csv` to weight the softmax cross entropy loss.

## Training

```
CUDA_VISIBLE_DEVICES=0 bash experiments/train.sh
```

You can specify which GPU you want to use by using `CUDA_VISIBLE_DEVICES` environment variable. Or if you directory use `train.py` instead of prepared training shell scripts in `experimetns` dir, you can easily specify the GPU ID by using `--gpu` argment.

### About train.sh

To use the preliminarily calculated coefficients to weight the softmax cross entropy in class-wise manner, add `--use_class_weight` option to the above command.

What the shell script `train.sh` do is a simple sequential training process. Once the first training for a most outer encoder-decoder pair, start the training for the next inner pair from the saved model state of the previous training process.

If you would like to change the training settings and hyper parameters, please see the output of `python train.py --help` to check the argments it can take.

## Prediction

Use `predict.py` to create prediction results on the test dataset. The below script executes the prediction script for all result dirs.

```
python experiments/batch_predict.py
```

## Evaluation

```
python experiments/batch_evaluate.py
```

# Results

The below table shows the evaluation results. Note that **STD** and **CW** stand for standardization (mean subtraction and stddev division) and class weight (per-class weighting for the softmax cross entropy during training), respectively.

| Model                        | Opt                 | Building | Tree | Sky | Car | SignSymbol | Road | Pedestrian | Fence | Pole | Pavement | Bicyclist | Class avg. | Global avg. |
|:----------------------------:|:-------------------:|:--------:|:----:|:---:|:---:|:----------:|:----:|:----------:|:-----:|:----:|:--------:|:---------:|:----------:|:-----------:|
| SegNet - 4 layer (from paper)  | L-BFGS              | 75.0 | 84.6 | 91.2 | 82.7 | 36.9 | 93.3 | 55.0 | 37.5 | 44.8 | 74.1 | 16.0 | 62.9 | 84.3 |
| chainer-segnet (No STD, No CW) | Adam (alpha=0.0001) | 86.3 | 82.7 | 96.0 | 82.4 | 40.7 | 93.2 | 43.3 | 40.8 | 23.9 | 76.9 | 24.1 | 57.5 | 86.2 |
| chainer-segnet (No STD)        | Adam (alpha=0.0001) | 69.0 | 79.4 | 90.5 | 90.5 | 51.2 | 92.5 | 75.8 | 51.7 | 61.8 | 82.8 | 59.1 | 67.0 | 82.1 |
| chainer-segnet                 | Adam (alpha=0.0001) |
# Reference

> Vijay Badrinarayanan, Alex Kendall and Roberto Cipolla "SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation." arXiv preprint arXiv:1511.00561, 2015\. [PDF](http://arxiv.org/abs/1511.00561)

## Official Implementation with Caffe

- [alexgkendall/SegNet-Tutorial](https://github.com/alexgkendall/SegNet-Tutorial)
- [alexgkendall/caffe-segnet](https://github.com/alexgkendall/caffe-segnet)
