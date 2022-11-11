#!/usr/bin/env bash
gdown http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz
tar -zxvf CUB_200_2011.tgz
python CUB2CRL.py