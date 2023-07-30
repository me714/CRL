## Fine-grained classification of surgical instruments combined with concept relation learning

_________________

This project is an official implementation of our method name CRL. Our method aims to address fine-grained segmentation of small samples of surgical instruments.

<p align="center">
<img src="https://github.com/me714/CRL/blob/depth/image/figure2.jpg" width="1100" align="center">
<img src="https://github.com/me714/CRL/blob/depth/image/figure3.jpg" width="1100" align="center">
</p>


### Dependencies
- python 3.7
- pytorch 1.7.1
- numpy 1.21.2
- h5py
- tensorboard
- easydict
- pillow

### Installation
1. Clone or download this code repository

```
git clone https://github.com/yourusername/your-repo.git
cd your-repo
```

2. Create and activate a virtual environment (optional, but highly recommended).
 
```
python -m venv venv
source venv/bin/activate

```
3. Installation of required dependencies
```
pip install -r requirements.txt

```

### Data Preparation

Before running our method, make sure that you have prepared the relevant dataset. The dataset should contain the following:

```
cd ./filelists/CUB
source ./download_CUB.sh
```



### How to Use

#### Training

Run the following command to train our method:

```
python ./train.py --dataset CUB --model Conv6NP --method comet --train_aug
```

##### Testing

Run the following command to test our method:

```
python ./test.py --dataset CUB --model Conv6NP --method comet --train_aug
```
### Results
<img src="https://github.com/me714/CRL/blob/depth/image/uTools_1690703898324.png" width="1100" align="center">
</p>

### Additional Notes
Our codebase is developed based on the [benchmark implementation](https://github.com/wyharveychen/CloserLookFewShot) from paper [A Closer Look at Few-shot Classification](https://openreview.net/pdf?id=HkxLXnAcFQ) and [COMET](https://github.com/snap-stanford/comet).
