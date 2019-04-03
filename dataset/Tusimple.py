import json
import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class Tusimple(Dataset):
    def __init__(self, path, image_set, transforms=None):
        super(Tusimple, self).__init__()
        assert image_set in ('train', 'val', 'test'), "image_set is not valid!"
        self.data_dir_path = path
        self.image_set = image_set
        self.transforms = transforms

        self.createIndex()

    def createIndex(self):
        self.img_list = []
        self.segLabel_list = []
        self.exist_list = []

        listfile = os.path.join(self.data_dir_path, "seg_label", "list", "list_gt.txt")
        with open(listfile) as f:
            for line in f:
                line = line.strip()
                l = line.split(" ")
                self.img_list.append(os.path.join(self.data_dir_path, l[0][1:]))  # l[0][1:]  get rid of the first '/' so as for os.path.join
                self.segLabel_list.append(os.path.join(self.data_dir_path, l[1][1:]))
                self.exist_list.append([int(x) for x in l[2:]])

    def __getitem__(self, idx):
        img = cv2.imread(self.img_list[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.image_set != 'test':
            segLabel = cv2.imread(self.segLabel_list[idx])[:, :, 0]
            exist = self.exist_list[idx]
        else:
            segLabel = None
            exist = None

        if self.transforms is not None:
            img, segLabel, exist = self.transforms(img, segLabel, exist)

        sample = {'img': img,
                  'segLabel': segLabel,
                  'exist': exist,
                  'img_name': self.img_list[idx]}
        return sample

    def __len__(self):
        return len(self.img_list)

    def generate_label(self):
        H, W = 720, 1280
        SEG_WIDTH = 30
        save_dir = "seg_label"

        os.makedirs(os.path.join(self.data_dir_path, save_dir, "list"), exist_ok=True)
        list_f = open(os.path.join(self.data_dir_path, save_dir, "list", "list_gt.txt"), "w")

        json_path = os.path.join(self.data_dir_path, "label_data.json")
        with open(json_path) as f:
            for line in f:
                label = json.loads(line)
                img_path = label['raw_file']
                seg_img = np.zeros((H, W, 3))
                list_str = [] # str to be written to list.txt

                # ---------- clean and sort lanes -------------
                _lanes = []
                slope = [] # identify 1st, 2nd, 3rd, 4th lane through slope
                for i in range(len(label['lanes'])):
                    l = [(x, y) for x, y in zip(label['lanes'][i], label['h_samples']) if x >= 0]
                    if (len(l)>1):
                        _lanes.append(l)
                        slope.append(np.arctan2(l[-1][1]-l[0][1], l[0][0]-l[-1][0]) / np.pi * 180)
                _lanes = [_lanes[i] for i in np.argsort(slope)]
                slope = [slope[i] for i in np.argsort(slope)]

                idx_1 = None
                idx_2 = None
                idx_3 = None
                idx_4 = None
                for i in range(len(slope)):
                    if slope[i]<=90:
                        idx_2 = i
                        idx_1 = i-1 if i>0 else None
                    else:
                        idx_3 = i
                        idx_4 = i+1 if i+1 < len(slope) else None
                        break
                lanes = []
                lanes.append([] if idx_1 is None else _lanes[idx_1])
                lanes.append([] if idx_2 is None else _lanes[idx_2])
                lanes.append([] if idx_3 is None else _lanes[idx_3])
                lanes.append([] if idx_4 is None else _lanes[idx_4])
                # ---------------------------------------------

                for i in range(len(lanes)):
                    coords = lanes[i]
                    if len(coords) < 4:
                        list_str.append('0')
                        continue
                    for j in range(len(coords)-1):
                        cv2.line(seg_img, coords[j], coords[j+1], (i+1, i+1, i+1), SEG_WIDTH//2)
                    list_str.append('1')

                seg_path = img_path.split("/")
                seg_path, img_name = os.path.join(self.data_dir_path, save_dir, seg_path[1], seg_path[2]), seg_path[3]
                os.makedirs(seg_path, exist_ok=True)
                seg_path = os.path.join(seg_path, img_name[:-3]+"png")
                cv2.imwrite(seg_path, seg_img)

                seg_path = "/".join([save_dir, *img_path.split("/")[1:3], img_name])
                if seg_path[0] != '/':
                    seg_path = '/' + seg_path
                if img_path[0] != '/':
                    img_path = '/' + img_path
                list_str.insert(0, seg_path)
                list_str.insert(0, img_path)
                list_str = " ".join(list_str) + "\n"
                list_f.write(list_str)

        list_f.close()

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


if __name__ == "__main__":
    data = Tusimple(r"E:\Autonomous car research\Driving_Dataset\tusimple")
    data.generate_label()
