## Fine-grained classification of surgical instruments combined with concept relation learning

_________________

This repo contains the reference source code in PyTorch of the CRL(concept relation learning) algorithm.

<p align="center">
<img src="https://github.com/me714/CRL/blob/depth/image/figure2.jpg" width="1100" align="center">
<img src="https://github.com/me714/CRL/blob/depth/image/figure3.jpg" width="1100" align="center">
</p>


### Requirements
- python 3.7
- pytorch 1.7.1
- numpy 1.21.2
- h5py
- tensorboard
- easydict
- pillow

##### Dataset

* Change directory to `./filelists/CUB`
* Run `source ./download_CUB.sh`


### Usage

##### Training

We provide an example here:

Run
```python ./train.py --dataset CUB --model Conv6NP --method comet --train_aug```

##### Testing

We provide an example here:

Run
```python ./test.py --dataset CUB --model Conv6NP --method comet --train_aug```


Our codebase is developed based on the [benchmark implementation](https://github.com/wyharveychen/CloserLookFewShot) from paper [A Closer Look at Few-shot Classification](https://openreview.net/pdf?id=HkxLXnAcFQ) and [COMET](https://github.com/snap-stanford/comet).
