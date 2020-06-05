from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import pickle
import os
import cv2
import scipy.io as sio
import torch


# 一次性读入所有数据
# nyu/test/1318_a=0.55_b=1.21.png
class train_DataSet(Dataset):
    def __init__(self, transform1, path=None):
        self.transform1 = transform1
        self.haze_path, self.gt_path, self.dehazy_path = path

        self.haze_data_list = os.listdir(self.haze_path)
        self.haze_data_list.sort(key=lambda x: int(x[:-18]))
        self.dehazy_data_list = os.listdir(self.dehazy_path)
        self.dehazy_data_list.sort(key=lambda x: int(x[:-18]))
        self.gt_data_list = os.listdir(self.gt_path)
        self.gt_data_list.sort(key=lambda x: int(x[:-4]))

        self.length = len(os.listdir(self.haze_path))
        self.haze_image_dict = {}
        self.dehazy_image_dict = {}
        self.gth_image_dict = {}
        # 读入数据
        print('starting read image data...')
        for i in range(len(self.haze_data_list)):
            name = self.haze_data_list[i][:-4]
            self.haze_image_dict[name] = cv2.imread(self.haze_path + name + '.png')
        print('starting read dehazy data...')
        for i in range(len(self.dehazy_data_list)):
            name = self.dehazy_data_list[i][:-4]
            self.dehazy_image_dict[name] = cv2.imread(self.dehazy_path + name + '.png')
        print('starting read GroundTruth data...')
        for i in range(len(self.gt_data_list)):
            name = self.gt_data_list[i][:-4]
            self.gth_image_dict[name] = cv2.imread(self.gt_path + name + '.png')

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        name = self.haze_data_list[idx][:-4]
        haze_image = self.haze_image_dict[name]
        dehazy_image = self.dehazy_image_dict[name]
        gt_image = self.gth_image_dict[name[:4]]

        if self.transform1:
            haze_image = self.transform1(haze_image)
            dehazy_image = self.transform1(dehazy_image)
            gt_image = self.transform1(gt_image)

        haze_image = haze_image.cuda()
        dehazy_image = dehazy_image.cuda()
        gt_image = gt_image.cuda()
        return haze_image, dehazy_image, gt_image


class val_DataSet(Dataset):
    def __init__(self, transform1, path=None):
        self.transform1 = transform1
        self.haze_path, self.gt_path = path

        self.haze_data_list = os.listdir(self.haze_path)
        self.haze_data_list.sort(key=lambda x: int(x[:-18]))
        self.gt_data_list = os.listdir(self.gt_path)
        self.gt_data_list.sort(key=lambda x: int(x[:-4]))

        self.length = len(os.listdir(self.haze_path))
        self.haze_image_dict = {}
        self.gth_image_dict = {}
        # 读入数据
        print('starting read image data...')
        for i in range(len(self.haze_data_list)):
            name = self.haze_data_list[i][:-4]
            self.haze_image_dict[name] = cv2.imread(self.haze_path + name + '.png')
        print('starting read GroundTruth data...')
        for i in range(len(self.gt_data_list)):
            name = self.gt_data_list[i][:-4]
            self.gth_image_dict[name] = cv2.imread(self.gt_path + name + '.png')

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        name = self.haze_data_list[idx][:-4]
        haze_image = self.haze_image_dict[name]
        gt_image = self.gth_image_dict[name[:4]]

        if self.transform1:
            haze_image = self.transform1(haze_image)
            gt_image = self.transform1(gt_image)

        haze_image = haze_image.cuda()
        gt_image = gt_image.cuda()
        return haze_image, gt_image


class test_DataSet(Dataset):
    def __init__(self, transform1, path=None):
        self.transform1 = transform1
        self.haze_path, self.gt_path = path
        self.haze_data_list = os.listdir(self.haze_path)
        self.haze_data_list.sort(key=lambda x: int(x[:-18]))
        self.gt_data_list = os.listdir(self.gt_path)
        self.gt_data_list.sort(key=lambda x: int(x[:-4]))

        self.length = len(os.listdir(self.haze_path))
        self.haze_image_dict = {}
        self.gth_image_dict = {}
        # 读入数据
        print('starting read image data...')
        for i in range(len(self.haze_data_list)):
            name = self.haze_data_list[i][:-4]
            self.haze_image_dict[name] = cv2.imread(self.haze_path + name + '.png')
        print('starting read GroundTruth data...')
        for i in range(len(self.gt_data_list)):
            name = self.gt_data_list[i][:-4]
            self.gth_image_dict[name] = cv2.imread(self.gt_path + name + '.png')

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        name = self.haze_data_list[idx][:-4]
        haze_image = self.haze_image_dict[name]
        gt_image = self.gth_image_dict[name[:4]]

        if self.transform1:
            haze_image = self.transform1(haze_image)
            gt_image = self.transform1(gt_image)

        haze_image = haze_image.cuda()
        gt_image = gt_image.cuda()
        return name, haze_image, gt_image

# if __name__ == '__main__':
