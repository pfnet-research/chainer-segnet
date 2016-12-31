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

The below table shows the evaluation results. Each column means:

- **Class weight**: It means the weight for softmax cross entropy. If class weight calculated from training labels using `lib/calc_mean.py`, it shows `Yes`. If a set of class weights copied from the original implementation (from a caffe protobuf file) are used, it shows `Original`. If no class weight is used, it shows `No`.
- **Standardization**: It means mean subtraction and stddev division.
- **Data Aug.**: Data augmentation by random rotation and flipping left/right, and random translation and scale jittering.
- **# conv channels**: The number of convolution filters used for all convolutional layers in the SegNet model.
- **End-to-End**: `Pairwise` means the way to train the SegNet in encoder-decoder pairwise manner. `Finetune` means that the model was finetuned after the pairwise training of encoder-decorder pairs in end-to-end manner. `End-to-End` means that the model was trained in end-to-end manner from the beggining to the end.

**Please find the more detailed results here: [`experiments/README.md`](https://github.com/mitmul/chainer-segnet/tree/master/experiments/README.md)**

| Model | Opt | Class weight | Standardization | Data Aug. | # conv channels | End-to-End | Class avg. | Global avg. |
|:-----:|:---:|:------------:|:---------------:|:---------:|:---------------:|:----------:|:-----------:|
| SegNet - 4 layer (from paper) | L-BFGS   | Original | ?   | ?   | 64  | Pairwise   | 62.9 | 84.3 |
| chainer-segnet | Adam (alpha=0.0001)     | Yes      | Yes | Yes | 128 | Pairwise   | **69.8** | 86.0 |
| chainer-segnet | Adam (alpha=0.0001)     | Original | Yes | Yes | 64  | Pairwise   | 68.6 | 82.2 |
| chainer-segnet | Adam (alpha=0.0001)     | Original | Yes | Yes | 64  | Finetune   | 68.5 | 83.3 |
| chainer-segnet | Adam (alpha=0.0001)     | Yes      | No  | Yes | 64  | Pairwise   | 68.0 | 82.3 |
| chainer-segnet | Adam (alpha=0.0001)     | Original | Yes | Yes | 128 | Pairwise   | 67.3 | 86.5 |
| chainer-segnet | Adam (alpha=0.0001)     | Yes      | Yes | Yes | 128 | Finetune   | 67.3 | 86.4 |
| chainer-segnet | Adam (alpha=0.0001)     | Yes      | No  | Yes | 64  | Finetune   | 66.9 | 83.5 |
| chainer-segnet | Adam (alpha=0.0001)     | Original | Yes | Yes | 128 | Finetune   | 66.3 | 86.2 |
| chainer-segnet | Adam (alpha=0.0001)     | Yes      | Yes | Yes | 64  | Finetune   | 65.5 | 82.9 |
| chainer-segnet | Adam (alpha=0.0001)     | Original | Yes | Yes | 64  | Finetune   | 65.1 | 80.5 |
| chainer-segnet | Adam (alpha=0.0001)     | Original | Yes | Yes | 64  | Pairwise   | 64.8 | 79.8 |
| chainer-segnet | MomentumSGD (lr=0.0001) | Yes      | Yes | Yes | 64  | Pairwise   | 64.8 | 76.9 |
| chainer-segnet | MomentumSGD (lr=0.0001) | Yes      | Yes | Yes | 64  | Finetune   | 64.7 | 79.8 |
| chainer-segnet | Adam (alpha=0.0001)     | Yes      | Yes | Yes | 64  | Pairwise   | 64.4 | 81.1 |
| chainer-segnet | Adam (alpha=0.0001)     | Yes      | Yes | Yes | 64  | End-to-End | 62.6 | 82.3 |
| chainer-segnet | Adam (alpha=0.0001)     | No       | No  | Yes | 64  | Pairwise   | 58.9 | **86.9** |
| chainer-segnet | Adam (alpha=0.0001)     | No       | Yes | Yes | 64  | Finetune   | 58.0 | 85.5 |
| chainer-segnet | Adam (alpha=0.0001)     | No       | No  | Yes | 64  | Finetune   | 57.2 | 87.0 |
| chainer-segnet | Adam (alpha=0.0001)     | No       | Yes | Yes | 64  | Pairwise   | 56.3 | 85.8 |
| chainer-segnet | Adam (alpha=0.0001)     | Yes      | Yes | No  | 64  | Pairwise   | 56.2 | 83.9 |
| chainer-segnet | Adam (alpha=0.0001)     | Yes      | Yes | No  | 64  | Finetune   | 54.1 | 83.3 |
| chainer-segnet | Adam (alpha=0.0001)     | Yes      | Yes | No  | 64  | End-to-End | 47.0 | 80.6 |

## Discussion

- Several models exceeded the accuracy described in the original paper.
- Larger number of channels leads better results.
- The original class weights seem to be better than the ones calculated using `lib/calc_mean.py` in this repository.
- Finetuning the model after enc-dec pairwise training improves global average accuracy but it decreases the class average accuracy in many cases.
- Pairwise (w/ or w/o finetuning) is almost always better than completely end-to-end training.
- Data augmentation is necessary.
- Standardization decreases both accuracy (class avg. and global avg.) in several cases.

# Reference

> Vijay Badrinarayanan, Alex Kendall and Roberto Cipolla "SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation." arXiv preprint arXiv:1511.00561, 2015\. [PDF](http://arxiv.org/abs/1511.00561)

## Official Implementation with Caffe

- [alexgkendall/SegNet-Tutorial](https://github.com/alexgkendall/SegNet-Tutorial)
- [alexgkendall/caffe-segnet](https://github.com/alexgkendall/caffe-segnet)
