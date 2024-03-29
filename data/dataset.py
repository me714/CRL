# This code is modified from https://github.com/facebookresearch/low-shot-shrink-hallucinate

import torch
from PIL import Image
import json
import numpy as np
import torchvision.transforms as transforms
import os
from torch.utils.data import Dataset
import random
import copy
import cv2
from .transforms import *

# this is a helper function to convert PIL image to byte tensor
identity = lambda x:x

# The code is a Python class called SimpleDataset that represents a dataset of images. The class takes as input a data file, image size, transform, target transform, and a boolean flag indicating whether the dataset is for training or not. The class reads the data file, which is expected to be in JSON format, and loads the image names and labels into memory. The __getitem__ method is used to retrieve an image and its corresponding label at a given index. The method reads the image from disk, applies some random transformations to it if the dataset is for training, and returns the transformed image and its label. The __len__ method returns the number of images in the dataset.
class SimpleDataset:
    def __init__(self, data_file, image_size, transform, target_transform=identity, is_train=True):
        with open(data_file, 'r') as f:
            self.meta = json.load(f)
        self.transform = transform
        self.target_transform = target_transform
        self.flip = is_train
        self.image_size = image_size
        self.is_train = is_train

    def __getitem__(self,i):
        image_path = os.path.join(self.meta['image_names'][i])
        data_numpy = cv2.imread(image_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)

        if data_numpy is None:
            raise ValueError('Fail to read {}'.format(image_path))

        r = 0
        c = np.array([data_numpy.shape[1], data_numpy.shape[0]]) // 2
        s = np.array([data_numpy.shape[1], data_numpy.shape[0]]) // 160

        if self.is_train:
            sf = 0.25
            rf = 30
            s = s * np.clip(np.random.randn()*sf + 1, 1 - sf, 1 + sf)
            r = np.clip(np.random.randn()*rf, -rf*2, rf*2) \
                if random.random() <= 0.6 else 0

            if self.flip and random.random() <= 0.5:
                data_numpy = data_numpy[:, ::-1, :]
                c[0] = data_numpy.shape[1] - c[0] - 1
            
        trans = get_affine_transform(c, s, r, [self.image_size, self.image_size])
        input = cv2.warpAffine(
            data_numpy,
            trans,
            (int(self.image_size), int(self.image_size)),
            flags=cv2.INTER_LINEAR)
        input = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)
        input = Image.fromarray(input.transpose((1,0,2)))

        if self.transform:
            input = self.transform(input)
        target = self.target_transform(self.meta['image_labels'][i])
        return input, target

    def __len__(self):
        return len(self.meta['image_names'])


class SetDataset:
    def __init__(self, data_file, batch_size, image_size, transform, is_train=True):
        with open(data_file, 'r') as f:
            self.meta = json.load(f)
 
        self.cl_list = np.unique(self.meta['image_labels']).tolist()

        self.sub_meta = {}
        for cl in self.cl_list:
            self.sub_meta[cl] = []

        if 'part' in self.meta:
            for x,y,z in zip(self.meta['image_names'],self.meta['image_labels'], self.meta['part']):
                self.sub_meta[y].append({'path':x, 'part': z})
        else:
            for x,y in zip(self.meta['image_names'],self.meta['image_labels']):
                self.sub_meta[y].append({'path':x})

        self.sub_dataloader = [] 
        sub_data_loader_params = dict(batch_size = batch_size,
                                  shuffle = True,
                                  num_workers = 0, #use main thread only or may receive multiple batches
                                  pin_memory = False)   

        if 'part' in self.meta:
            for cl in self.cl_list:
                sub_dataset = SubPartsDataset(self.sub_meta[cl], cl, image_size, transform = transform, is_train = is_train )
                self.sub_dataloader.append( torch.utils.data.DataLoader(sub_dataset, **sub_data_loader_params) )
        else:     
            for cl in self.cl_list:
                sub_dataset = SubDataset(self.sub_meta[cl], cl, transform = transform )
                self.sub_dataloader.append( torch.utils.data.DataLoader(sub_dataset, **sub_data_loader_params) )

    def __getitem__(self,i):
        return next(iter(self.sub_dataloader[i]))

    def __len__(self):
        return len(self.cl_list)

class EpisodicBatchSampler(object):
    def __init__(self, n_classes, n_way, n_episodes):
        self.n_classes = n_classes
        self.n_way = n_way
        self.n_episodes = n_episodes

    def __len__(self):
        return self.n_episodes

    def __iter__(self):
        for i in range(self.n_episodes):
            yield torch.randperm(self.n_classes)[:self.n_way]

class SubDataset:
    def __init__(self, sub_meta, cl, transform=transforms.ToTensor(), target_transform=identity):
        self.sub_meta = sub_meta
        self.cl = cl 
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self,i):
        #print( '%d -%d' %(self.cl,i))
        image_path = os.path.join(self.sub_meta[i]['path'])
        img = Image.open(image_path).convert('RGB')
        img = self.transform(img)
        target = self.target_transform(self.cl)
        return img, target

    def __len__(self):
        return len(self.sub_meta)

class SubPartsDataset(Dataset):

    def __init__(self, sub_meta, cl, image_size, transform=transforms.ToTensor(), target_transform=identity, is_train=True):
        self.num_joints = 4

        self.is_train = is_train
        self.sub_meta = sub_meta
        self.cl = cl 
        self.transform = transform
        self.target_transform = target_transform

        self.flip = is_train

        self.image_size = image_size
       
        self.transform = transform
        self.target_transform = target_transform


    def __len__(self,):
        return len(self.sub_meta)

    def __getitem__(self, idx):
        image_file = os.path.join(self.sub_meta[idx]['path'])
        
        data_numpy = cv2.imread(image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)

        if data_numpy is None:
            raise ValueError('Fail to read {}'.format(image_file))

        joints_vis = self.sub_meta[idx]['part']
        joints_vis = np.array(joints_vis)

        r = 0
        c = np.array([data_numpy.shape[1], data_numpy.shape[0]]) // 2
        s = np.array([data_numpy.shape[1], data_numpy.shape[0]]) // 160

        if self.is_train:
            sf = 0.25
            rf = 30
            s = s * np.clip(np.random.randn()*sf + 1, 1 - sf, 1 + sf)
            r = np.clip(np.random.randn()*rf, -rf*2, rf*2) if random.random() <= 0.6 else 0

            if self.flip and random.random() <= 0.5:
                data_numpy = data_numpy[:, ::-1, :]
                for i in range(self.num_joints):
                    if joints_vis[i, 2] > 0.0:
                        joints_vis[i, 0] = data_numpy.shape[1] - joints_vis[i, 0]
                c[0] = data_numpy.shape[1] - c[0] - 1
            
        trans = get_affine_transform(c, s, r, [self.image_size, self.image_size])
        input = cv2.warpAffine(
            data_numpy,
            trans,
            (int(self.image_size), int(self.image_size)),
            flags=cv2.INTER_LINEAR)
        input = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)
        input = Image.fromarray(input.transpose((1,0,2)))

        for i in range(self.num_joints):
            if joints_vis[i, 2] > 0.0:
                joints_vis[i, 0:2] = affine_transform(joints_vis[i, 0:2], trans)

        if self.transform:
            input = self.transform(input)

        target = self.target_transform(self.cl)

        joints_vis = self.target_transform(joints_vis)

        return input, target, joints_vis