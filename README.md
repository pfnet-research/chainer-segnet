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

The below table shows the evaluation results. Note that **STD** and **CW** stand for standardization (mean subtraction and stddev division) and class weight (per-class weighting for the softmax cross entropy during training), respectively. **DA** stands for data augmentation (random roatation, fliping LR, shift jittering, and scale jittering). `lr=0.01dc` means the learning rate was dropped starting from 0.01. See the actual lr schedule in `experiments/train_end2end_msgd.sh`.

| Model                           | Opt                         | Building | Tree | Sky | Car | SignSymbol | Road | Pedestrian | Fence | Pole | Pavement | Bicyclist | Class avg. | Global avg. | IoU |
|:-------------------------------:|:---------------------------:|:--------:|:----:|:---:|:---:|:----------:|:----:|:----------:|:-----:|:----:|:--------:|:---------:|:----------:|:-----------:|:---:|
| SegNet - 4 layer (from paper)   | L-BFGS                      |   75.0   | **84.6** |   91.2   |   82.7   |   36.9   |   93.3   |   55.0   |   37.5   |   44.8   |   74.1   |   16.0   |   62.9   |   84.3   | N/A      |
| chainer-segnet                  | Adam (alpha=0.0001)         |   76.6   |   75.3   |   93.9   | **88.7** |   51.5   |   91.6   |   77.5   |   53.1   |   57.2   |   73.7   |   46.8   |   65.5   |   82.9   |   47.3   |
| chainer-segnet                  | MomentumSGD (lr=0.0001)     |   72.6   |   64.3   |   93.9   |   88.2   |   52.1   |   90.0   | **78.3** | **58.1** |   55.8   |   69.5   | **53.0** |   64.7   |   79.8   |   43.4   |
| chainer-segnet (No DA)          | Adam (alpha=0.0001)         |   85.9   |   64.5   |   95.3   |   77.8   |   19.5   |   96.0   |   45.5   |   39.3   |   31.6   |   67.0   |   26.6   |   54.1   |   83.3   |   43.7   |
| chainer-segnet (No STD)         | Adam (alpha=0.0001)         |   74.4   |   75.8   |   94.1   |   88.6   | **60.2** |   90.8   |   74.0   |   52.5   | **59.6** | **85.9** |   46.5   | **66.9** |   83.5   |   47.6   |
| chainer-segnet (No CW)          | Adam (alpha=0.0001)         |   88.7   |   73.4   | **96.6** |   82.4   |   42.9   | **96.8** |   45.4   |   35.4   |   29.7   |   63.1   |   41.3   |   58.0   |   85.5   |   48.4   |
| chainer-segnet (No STD, No CW)  | Adam (alpha=0.0001)         | **90.2** |   81.0   |   96.4   |   87.1   |   28.2   |   96.1   |   47.3   |   33.5   |   25.6   |   68.5   |   32.1   |   57.2   | **87.0** | **49.2** |
| chainer-segnet (End2End)        | Adam (alpha=0.0001)         |   77.6   |   68.7   |   92.5   |   84.6   |   47.3   |   89.5   |   74.1   |   43.0   |   54.7   |   85.1   |   33.6   |   62.6   |   82.3   |   45.8   |
| chainer-segnet (End2End)        | MomentumSGD (lr=0.01dc)     |   75.3   |   67.7   |   92.3   |   86.8   |   53.6   |   93.7   |   62.9   |   29.5   |   51.2   |   68.0   |   41.7   |   60.2   |   80.9   |   43.9   |
| chainer-segnet (End2End, No DA) | Adam (alpha=0.0001)         |   86.8   |   56.9   |   93.4   |   72.4   |   9.60   |   95.5   |   24.4   |   25.2   |   24.4   |   58.5   |   16.8   |   47.0   |   80.6   |   38.4   |
| chainer-segnet (End2End, No STD, No CW) | Adam (alpha=0.0001) |   89.5   |   74.8   |   95.9   |   88.5   |   27.3   |   95.4   |   44.3   |   28.8   |   28.2   |   76.4   |   37.9   |   57.3   |   86.6   |   48.7   |

# Reference

> Vijay Badrinarayanan, Alex Kendall and Roberto Cipolla "SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation." arXiv preprint arXiv:1511.00561, 2015\. [PDF](http://arxiv.org/abs/1511.00561)

## Official Implementation with Caffe

- [alexgkendall/SegNet-Tutorial](https://github.com/alexgkendall/SegNet-Tutorial)
- [alexgkendall/caffe-segnet](https://github.com/alexgkendall/caffe-segnet)
