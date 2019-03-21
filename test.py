import argparse
import json
import os

import torch
from torch.utils.data import DataLoader

from config import *
from dataset.CULane import CULane
from net import SCNN

from utils.transforms import *
from utils.prob2lines import getLane

# ------------ config ------------
exp_dir = "./experiments/exp0"

with open(os.path.join(exp_dir, "cfg.json")) as f:
    exp_cfg = json.load(f)

device = torch.device(exp_cfg['device'])


def split_path(path):
    """split path tree into list"""
    folders = []
    while True:
        path, folder = os.path.split(path)
        if folder != "":
            folders.insert(0, folder)
        else:
            if path != "":
                folders.insert(0, path)
            break
    return folders


# ------------ data and model ------------
transform_val = Compose(Resize((800, 288)), ToTensor(),
                        Normalize(mean=(0.3598, 0.3653, 0.3662), std=(0.2573, 0.2663, 0.2756)))
val_dataset = CULane(CULane_path, "val", transform_val)
val_loader = DataLoader(val_dataset, batch_size=8, collate_fn=val_dataset.collate, num_workers=4)

net = SCNN(pretrained=False)
save_name = os.path.join(exp_dir, 'best.pth')
save_dict = torch.load(save_name, map_location='cpu')
net.load_state_dict(save_dict['net'])
net.to(device)
net.eval()


# ------------ test ------------
out_path = os.path.join(exp_dir, "coord_output")
if not os.path.exists(out_path):
    os.mkdir(out_path)

with torch.no_grad():
    for batch_idx, sample in enumerate(val_loader):
        img = sample['img'].to(device)
        img_name = sample['img_name']

        seg_pred, exist_pred = net(img)[:2]
        seg_pred = seg_pred.detach().cpu().numpy()
        exist_pred = exist_pred.detach().cpu().numpy()

        for b in range(len(seg_pred)):
            seg = seg_pred[b]
            exist = [1 if exist_pred[b, i] > 0.5 else 0 for i in range(4)]
            lane_coords = getLane.prob2lines(seg, exist, resize_shape=(590, 1640), y_px_gap=20, pts=18)

            path_tree = split_path(img_name[b])
            save_dir, save_name = path_tree[-3:-1], path_tree[-1]
            save_dir = os.path.join(out_path, *save_dir)
            save_name = save_name[:-3] + "lines.txt"
            save_name = os.path.join(save_dir, save_name)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            with open(save_name, "w") as f:
                for l in lane_coords:
                    for (x, y) in l:
                        print("{} {}".format(x, y), end=" ", file=f)
                    print(file=f)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", "-d", type=str)
    args = parser.parse_args()
    return args
