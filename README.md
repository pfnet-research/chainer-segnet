# SegNet
SegNet implementation &amp; experiments written in Chainer

## Requirements

- Python 2.7.12 (Python 3.5.1+ somehow stacks at multiprocessing)
- Chainer 1.17.0+
- scikit-learn 0.17.1
- NumPy 1.11.0+
- OpenCV 3.1.0
    - `conda install -c https://conda.binstar.org/menpo opencv3`

### For debug

```
conda install -c conda-forge gdb
CFLAGS='-Wall -O0 -g' python setup.py install
```

## Download

```
wget https://dl.dropboxusercontent.com/u/2498135/segnet/CamVid.tar.gz
tar zxvf CamVid.tar.gz; rm -rf CamVid.tar.gz; mv CamVid data
```

## Training

```
python train.py --gpus 0 --batchsize 16 --rotate --fliplr
```

To use the given coefficients to weight the softmax loss class-wise, add `--use_class_weights` option to the above command.
