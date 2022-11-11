## Fine-grained classification of surgical instruments combined with concept relation learning

_________________

This repo contains the reference source code in PyTorch of the CRL(concept relation learning) algorithm.

<p align="center">
<img src="https://github.com/snap-stanford/comet/blob/master/img/COMET_model.png" width="1100" align="center">
</p>

### Dependencies

The code is built with following libraries:

- [PyTorch](https://pytorch.org/) 1.5
- [anndata](https://icb-anndata.readthedocs-hosted.com/en/stable/anndata.AnnData.html)
- [scanpy](https://icb-scanpy.readthedocs-hosted.com/en/stable/)
- json
- [wandb](https://www.wandb.com/)

### Getting started

##### CUB dataset
* Change directory to `./filelists/CUB`
* Run `source ./download_CUB.sh`

##### Tabula Muris dataset
* Change directory to `./filelists/tabula_muris`
* Run `source ./download_TM.sh`

### Usage

##### Training

We provide an example here:

Run
```python ./train.py --dataset CUB --model Conv4NP --method comet --train_aug```

##### Testing

We provide an example here:

Run
```python ./test.py --dataset CUB --model Conv4NP --method comet --train_aug```

### Tabula Muris benchmark

If you would like to test your algorithm on the new benchmark dataset introduced in our work, you can download the data as described above or directly at [http://snap.stanford.edu/comet/data/tabula-muris-comet.zip](http://snap.stanford.edu/comet/data/tabula-muris-comet).

Dataset needs to be preprocessed using [preprocess.py](https://github.com/snap-stanford/comet/blob/master/TM/data/preprocess.py). Train/test/validation splits are available in [load_tabula_muris](https://github.com/snap-stanford/comet/blob/master/TM/data/dataset.py).

Running this code requires [anndata](https://icb-anndata.readthedocs-hosted.com/en/stable/anndata.AnnData.html) and [scanpy](https://icb-scanpy.readthedocs-hosted.com/en/stable/) libraries.

### Citing

If you find our code useful, please consider citing:

```
@inproceedings{
    cao2021concept,
    title={Concept Learners for Few-Shot Learning},
    author={Cao, Kaidi and Brbi\'c, Maria and Leskovec, Jure},
    booktitle={International Conference on Learning Representations (ICLR)},
    year={2021},
}
```

Our codebase is developed based on the [benchmark implementation](https://github.com/wyharveychen/CloserLookFewShot) from paper [A Closer Look at Few-shot Classification](https://openreview.net/pdf?id=HkxLXnAcFQ).


Tabula Muris benchmark is developed based on the mouse aging cell atlas from paper [https://www.nature.com/articles/s41586-020-2496-1](https://www.nature.com/articles/s41586-020-2496-1).