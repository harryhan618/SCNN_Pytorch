import cv2
import os
import numpy as np

import torch
from torch.utils.data import Dataset


class CULane(Dataset):
    def __init__(self, path, image_set, transforms=None):
        super(CULane, self).__init__()
        assert image_set in ('train', 'val', 'test'), "image_set is not valid!"
        self.data_dir_path = path
        self.image_set = image_set
        self.transforms = transforms

        if image_set != 'test':
            self.createIndex()
        else:
            self.createIndex_test()


    def createIndex(self):
        listfile = os.path.join(self.data_dir_path, "list", "{}_gt.txt".format(self.image_set))

        self.img_list = []
        self.segLabel_list = []
        self.exist_list = []
        with open(listfile) as f:
            for line in f:
                line = line.strip()
                l = line.split(" ")
                self.img_list.append(os.path.join(self.data_dir_path, l[0][1:]))   # l[0][1:]  get rid of the first '/' so as for os.path.join
                self.segLabel_list.append(os.path.join(self.data_dir_path, l[1][1:]))
                self.exist_list.append([int(x) for x in l[2:]])

    def createIndex_test(self):
        listfile = os.path.join(self.data_dir_path, "list", "{}.txt".format(self.image_set))

        self.img_list = []
        with open(listfile) as f:
            for line in f:
                line = line.strip()
                self.img_list.append(os.path.join(self.data_dir_path, line[1:]))  # l[0][1:]  get rid of the first '/' so as for os.path.join

    def __getitem__(self, idx):
        img = cv2.imread(self.img_list[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.image_set != 'test':
            segLabel = cv2.imread(self.segLabel_list[idx])[:, :, 0]
            exist = np.array(self.exist_list[idx])
        else:
            segLabel = None
            exist = None

        sample = {'img': img,
                  'segLabel': segLabel,
                  'exist': exist,
                  'img_name': self.img_list[idx]}
        if self.transforms is not None:
            sample = self.transforms(sample)
        return sample

    def __len__(self):
        return len(self.img_list)

    @staticmethod
    def collate(batch):
        if isinstance(batch[0]['img'], torch.Tensor):
            img = torch.stack([b['img'] for b in batch])
        else:
            img = [b['img'] for b in batch]

        if batch[0]['segLabel'] is None:
            segLabel = None
            exist = None
        elif isinstance(batch[0]['segLabel'], torch.Tensor):
            segLabel = torch.stack([b['segLabel'] for b in batch])
            exist = torch.stack([b['exist'] for b in batch])
        else:
            segLabel = [b['segLabel'] for b in batch]
            exist = [b['exist'] for b in batch]

        samples = {'img': img,
                  'segLabel': segLabel,
                  'exist': exist,
                  'img_name': [x['img_name'] for x in batch]}

        return samples