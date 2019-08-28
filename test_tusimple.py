import argparse
import json
import os

import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import dataset
from config import *
from model import SCNN
from utils.prob2lines import getLane
from utils.transforms import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str, default="./experiments/exp0")
    args = parser.parse_args()
    return args


# ------------ config ------------
args = parse_args()
exp_dir = args.exp_dir
exp_name = exp_dir.split('/')[-1]

with open(os.path.join(exp_dir, "cfg.json")) as f:
    exp_cfg = json.load(f)
resize_shape = tuple(exp_cfg['dataset']['resize_shape'])
device = torch.device('cuda')


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
# # CULane mean, std
# mean=(0.3598, 0.3653, 0.3662)
# std=(0.2573, 0.2663, 0.2756)
# Imagenet mean, std
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
transform = Compose(Resize(resize_shape), ToTensor(),
                    Normalize(mean=mean, std=std))
dataset_name = exp_cfg['dataset'].pop('dataset_name')
Dataset_Type = getattr(dataset, dataset_name)
test_dataset = Dataset_Type(Dataset_Path['Tusimple'], "test", transform)
test_loader = DataLoader(test_dataset, batch_size=32, collate_fn=test_dataset.collate, num_workers=4)

net = SCNN(input_size=resize_shape, pretrained=False)
save_name = os.path.join(exp_dir, exp_dir.split('/')[-1] + '_best.pth')
save_dict = torch.load(save_name, map_location='cpu')
print("\nloading", save_name, "...... From Epoch: ", save_dict['epoch'])
net.load_state_dict(save_dict['net'])
net = torch.nn.DataParallel(net.to(device))
net.eval()

# ------------ test ------------
out_path = os.path.join(exp_dir, "coord_output")
evaluation_path = os.path.join(exp_dir, "evaluate")
if not os.path.exists(out_path):
    os.mkdir(out_path)
if not os.path.exists(evaluation_path):
    os.mkdir(evaluation_path)
dump_to_json = []

progressbar = tqdm(range(len(test_loader)))
with torch.no_grad():
    for batch_idx, sample in enumerate(test_loader):
        img = sample['img'].to(device)
        img_name = sample['img_name']

        seg_pred, exist_pred = net(img)[:2]
        seg_pred = F.softmax(seg_pred, dim=1)
        seg_pred = seg_pred.detach().cpu().numpy()
        exist_pred = exist_pred.detach().cpu().numpy()

        for b in range(len(seg_pred)):
            seg = seg_pred[b]
            exist = [1 if exist_pred[b, i] > 0.5 else 0 for i in range(4)]
            lane_coords = getLane.prob2lines_tusimple(seg, exist, resize_shape=(720, 1280), y_px_gap=10, pts=56)
            for i in range(len(lane_coords)):
                lane_coords[i] = sorted(lane_coords[i], key=lambda pair: pair[1])

            path_tree = split_path(img_name[b])
            save_dir, save_name = path_tree[-3:-1], path_tree[-1]
            save_dir = os.path.join(out_path, *save_dir)
            save_name = save_name[:-3] + "lines.txt"
            save_name = os.path.join(save_dir, save_name)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)

            with open(save_name, "w") as f:
                for l in lane_coords:
                    for (x, y) in l:
                        print("{} {}".format(x, y), end=" ", file=f)
                    print(file=f)

            json_dict = {}
            json_dict['lanes'] = []
            json_dict['h_sample'] = []
            json_dict['raw_file'] = os.path.join(*path_tree[-4:])
            json_dict['run_time'] = 0
            for l in lane_coords:
                if len(l) == 0:
                    continue
                json_dict['lanes'].append([])
                for (x, y) in l:
                    json_dict['lanes'][-1].append(int(x))
            for (x, y) in lane_coords[0]:
                json_dict['h_sample'].append(y)
            dump_to_json.append(json.dumps(json_dict))

        progressbar.update(1)
progressbar.close()

with open(os.path.join(out_path, "predict_test.json"), "w") as f:
    for line in dump_to_json:
        print(line, end="\n", file=f)

# ---- evaluate ----
from utils.lane_evaluation.tusimple.lane import LaneEval

eval_result = LaneEval.bench_one_submit(os.path.join(out_path, "predict_test.json"),
                                        "/home/lion/Dataset/tusimple/test_label.json")
print(eval_result)
with open(os.path.join(evaluation_path, "evaluation_result.txt"), "w") as f:
    print(eval_result, file=f)
