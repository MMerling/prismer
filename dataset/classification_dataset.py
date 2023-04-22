# Copyright (c) 2023, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://github.com/NVlabs/prismer/blob/main/LICENSE

import glob
from torch.utils.data import Dataset
from dataset.utils import *
import pandas as pd
import numpy as np


class Classification(Dataset):
    def __init__(self, config, train, test=False):
        self.data_path = config['data_path']
        self.label_path = config['label_path']
        self.experts = config['experts']
        self.dataset = config['dataset']
        self.shots = config['shots']
        self.prefix = config['prefix']

        self.train = train
        self.test = test
        self.transform = Transform(resize_resolution=config['image_resolution'], scale_size=[0.5, 1.0], train=True)

        if train:
            data_folders = glob.glob(f'{self.data_path}/img/')
            self.json_list = pd.read_json(open(f'{self.data_path}/' + 'train.jsonl'), lines=True)
            self.answer_list = ["normal", "hateful"]
            # self.data_list = [{'image': data} for f in data_folders for data in glob.glob(f + '*.png')[:self.shots]]
            self.data_list = [{'image': data} for data in self.json_list["img"].values]
        elif test:
            self.json_list = pd.read_json(open(f'{self.data_path}/' + 'test_combined.jsonl'), lines=True)
            self.answer_list = ["normal", "hateful"]
            # self.data_list = [{'image': data} for f in data_folders for data in glob.glob(f + '*.png')[:self.shots]]
            self.data_list = [{'image': data} for data in self.json_list["img"].values]
        else:
            data_folders = glob.glob(f'{self.data_path}/img/')
            # self.data_list = [{'image': data} for f in data_folders for data in glob.glob(f + '*.png')]
            self.json_list = pd.read_json(open(f'{self.data_path}/' + 'dev_combined.jsonl'), lines=True)

            self.answer_list = ["normal", "hateful"]
            self.data_list = [{'image': data} for data in self.json_list["img"].values]
        self.json_list["hateful"] = np.where(self.json_list["label"] == 0, "normal", "hateful")
        # self.answer_list["hateful"] = np.where(self.answer_list["label"] == 0, "normal", "hateful")
        # print(self.data_list)
        # print(self.answer_list)
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        img_path = self.data_list[index]['image']
        if self.train:
            img_path_split = img_path.split('/')
            img_name = img_path_split[-2] + '/' + img_path_split[-1]
            class_name = img_path_split[-2]
            image, labels, labels_info = get_expert_labels(self.data_path, self.label_path, img_name, self.dataset, self.experts, True)
        elif self.test:
            img_path_split = img_path.split('/')
            img_name = img_path_split[-2] + '/' + img_path_split[-1]
            class_name = img_path_split[-2]
            image, labels, labels_info = get_expert_labels(self.data_path, self.label_path, img_name, self.dataset,
                                                           self.experts, True)
        else:
            img_path_split = img_path.split('/')
            img_name = img_path_split[-2] + '/' + img_path_split[-1]
            class_name = img_path_split[-2]
            image, labels, labels_info = get_expert_labels(self.data_path, self.label_path, img_name, self.dataset, self.experts, True)

        experts = self.transform(image, labels)
        experts = post_label_process(experts, labels_info)

        if self.train:
            caption = self.prefix + ' ' + self.json_list[self.json_list["img"]==img_name]["hateful"].to_list()[0]
            return experts, caption
        else:
            # caption = self.prefix + ' ' + self.json_list[self.json_list["img"] == img_name]["hateful"].to_list()[0]
            return experts, self.json_list[self.json_list["img"] == img_name]["label"].to_list()[0]





# import os
# import glob
#
# data_path = '/Users/shikunliu/Documents/dataset/mscoco/mscoco'
#
# data_folders = glob.glob(f'{data_path}/*/')
# data_list = [data for f in data_folders for data in glob.glob(f + '*.jpg')]


