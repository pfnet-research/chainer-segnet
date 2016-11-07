# SegNet
SegNet implementation &amp; experiments written in Chainer

## Requirements

- Python
- Chainer 1.17.0+
- scikit-learn
- NumPy
- OpenCV

## Download

```
wget https://dl.dropboxusercontent.com/u/2498135/segnet/CamVid.tar.gz
tar zxvf CamVid.tar.gz; rm -rf CamVid.tar.gz; mv CamVid data
```

## Training

```
python train.py --gpus 0,1,2,3
```

To use the given coefficients to weight the softmax loss class-wise, add `--use_class_weights` option to the above command.
