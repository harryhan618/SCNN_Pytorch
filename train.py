import argparse
import json
import os
import shutil
import time

import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import *
import dataset
from model import SCNN
from utils.tensorboard import TensorBoard
from utils.transforms import *
from utils.lr_scheduler import PolyLR


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str, default="./experiments/exp0")
    parser.add_argument("--resume", "-r", action="store_true")
    args = parser.parse_args()
    return args
args = parse_args()

# ------------ config ------------
exp_dir = args.exp_dir
while exp_dir[-1]=='/':
    exp_dir = exp_dir[:-1]
exp_name = exp_dir.split('/')[-1]

with open(os.path.join(exp_dir, "cfg.json")) as f:
    exp_cfg = json.load(f)
resize_shape = tuple(exp_cfg['dataset']['resize_shape'])

device = torch.device(exp_cfg['device'])
tensorboard = TensorBoard(exp_dir)

# ------------ train data ------------
# # CULane mean, std
# mean=(0.3598, 0.3653, 0.3662)
# std=(0.2573, 0.2663, 0.2756)
# Imagenet mean, std
mean=(0.485, 0.456, 0.406)
std=(0.229, 0.224, 0.225)
transform_train = Compose(Resize(resize_shape), Rotation(2), ToTensor(),
                          Normalize(mean=mean, std=std))
dataset_name = exp_cfg['dataset'].pop('dataset_name')
Dataset_Type = getattr(dataset, dataset_name)
train_dataset = Dataset_Type(Dataset_Path[dataset_name], "train", transform_train)
train_loader = DataLoader(train_dataset, batch_size=exp_cfg['dataset']['batch_size'], shuffle=True, collate_fn=train_dataset.collate, num_workers=8)

# ------------ val data ------------
transform_val_img = Resize(resize_shape)
transform_val_x = Compose(ToTensor(), Normalize(mean=mean, std=std))
transform_val = Compose(transform_val_img, transform_val_x)
val_dataset = Dataset_Type(Dataset_Path[dataset_name], "val", transform_val)
val_loader = DataLoader(val_dataset, batch_size=8, collate_fn=val_dataset.collate, num_workers=4)

# ------------ preparation ------------
net = SCNN(resize_shape, pretrained=True)
net = net.to(device)
net = torch.nn.DataParallel(net)

optimizer = optim.SGD(net.parameters(), **exp_cfg['optim'])
lr_scheduler = PolyLR(optimizer, 0.9, **exp_cfg['lr_scheduler'])
best_val_loss = 1e6


def train(epoch):
    print("Train Epoch: {}".format(epoch))
    net.train()
    train_loss = 0
    train_loss_seg = 0
    train_loss_exist = 0
    progressbar = tqdm(range(len(train_loader)))

    for batch_idx, sample in enumerate(train_loader):
        img = sample['img'].to(device)
        segLabel = sample['segLabel'].to(device)
        exist = sample['exist'].to(device)

        optimizer.zero_grad()
        seg_pred, exist_pred, loss_seg, loss_exist, loss = net(img, segLabel, exist)
        if isinstance(net, torch.nn.DataParallel):
            loss_seg = loss_seg.sum()
            loss_exist = loss_exist.sum()
            loss = loss.sum()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        iter_idx = epoch * len(train_loader) + batch_idx
        train_loss = loss.item()
        train_loss_seg = loss_seg.item()
        train_loss_exist = loss_exist.item()
        progressbar.set_description("batch loss: {:.3f}".format(loss.item()))
        progressbar.update(1)

        lr = optimizer.param_groups[0]['lr']
        tensorboard.scalar_summary(exp_name + "/train_loss", train_loss, iter_idx)
        tensorboard.scalar_summary(exp_name + "/train_loss_seg", train_loss_seg, iter_idx)
        tensorboard.scalar_summary(exp_name + "/train_loss_exist", train_loss_exist, iter_idx)
        tensorboard.scalar_summary(exp_name + "/learning_rate", lr, iter_idx)

    progressbar.close()
    tensorboard.writer.flush()

    if epoch % 1 == 0:
        save_dict = {
            "epoch": epoch,
            "net": net.module.state_dict() if isinstance(net, torch.nn.DataParallel) else net.state_dict(),
            "optim": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "best_val_loss": best_val_loss
        }
        save_name = os.path.join(exp_dir, exp_name + '.pth')
        torch.save(save_dict, save_name)
        print("model is saved: {}".format(save_name))

    print("------------------------\n")


def val(epoch):
    global best_val_loss

    print("Val Epoch: {}".format(epoch))

    net.eval()
    val_loss = 0
    val_loss_seg = 0
    val_loss_exist = 0
    progressbar = tqdm(range(len(val_loader)))

    with torch.no_grad():
        for batch_idx, sample in enumerate(val_loader):
            img = sample['img'].to(device)
            segLabel = sample['segLabel'].to(device)
            exist = sample['exist'].to(device)

            seg_pred, exist_pred, loss_seg, loss_exist, loss = net(img, segLabel, exist)
            if isinstance(net, torch.nn.DataParallel):
                loss_seg = loss_seg.sum()
                loss_exist = loss_exist.sum()
                loss = loss.sum()

            # visualize validation every 5 frame, 50 frames in all
            gap_num = 5
            if batch_idx%gap_num == 0 and batch_idx < 50 * gap_num:
                origin_imgs = []
                seg_pred = seg_pred.detach().cpu().numpy()
                exist_pred = exist_pred.detach().cpu().numpy()

                for b in range(len(img)):
                    img_name = sample['img_name'][b]
                    img = cv2.imread(img_name)
                    img = transform_val_img({'img': img})['img']

                    lane_img = np.zeros_like(img)
                    color = np.array([[255, 125, 0], [0, 255, 0], [0, 0, 255], [0, 255, 255]], dtype='uint8')

                    coord_mask = np.argmax(seg_pred[b], axis=0)
                    for i in range(0, 4):
                        if exist_pred[b, i] > 0.5:
                            lane_img[coord_mask==(i+1)] = color[i]
                    img = cv2.addWeighted(src1=lane_img, alpha=0.8, src2=img, beta=1., gamma=0.)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    lane_img = cv2.cvtColor(lane_img, cv2.COLOR_BGR2RGB)
                    cv2.putText(lane_img, "{}".format([1 if exist_pred[b, i]>0.5 else 0 for i in range(4)]), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2)
                    origin_imgs.append(img)
                    origin_imgs.append(lane_img)
                tensorboard.image_summary("img_{}".format(batch_idx), origin_imgs, epoch)

            val_loss += loss.item()
            val_loss_seg += loss_seg.item()
            val_loss_exist += loss_exist.item()

            progressbar.set_description("batch loss: {:.3f}".format(loss.item()))
            progressbar.update(1)

    progressbar.close()
    iter_idx = (epoch + 1) * len(train_loader)  # keep align with training process iter_idx
    tensorboard.scalar_summary("val_loss", val_loss, iter_idx)
    tensorboard.scalar_summary("val_loss_seg", val_loss_seg, iter_idx)
    tensorboard.scalar_summary("val_loss_exist", val_loss_exist, iter_idx)
    tensorboard.writer.flush()

    print("------------------------\n")
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        save_name = os.path.join(exp_dir, exp_name + '.pth')
        copy_name = os.path.join(exp_dir, exp_name + '_best.pth')
        shutil.copyfile(save_name, copy_name)


def main():
    global best_val_loss
    if args.resume:
        save_dict = torch.load(os.path.join(exp_dir, exp_name + '.pth'))
        if isinstance(net, torch.nn.DataParallel):
            net.module.load_state_dict(save_dict['net'])
        else:
            net.load_state_dict(save_dict['net'])
        optimizer.load_state_dict(save_dict['optim'])
        lr_scheduler.load_state_dict(save_dict['lr_scheduler'])
        start_epoch = save_dict['epoch'] + 1
        best_val_loss = save_dict.get("best_val_loss", 1e6)
    else:
        start_epoch = 0

    exp_cfg['MAX_EPOCHES'] = int(np.ceil(exp_cfg['lr_scheduler']['max_iter'] / len(train_loader)))
    for epoch in range(start_epoch, exp_cfg['MAX_EPOCHES']):
        train(epoch)
        if epoch % 1 == 0:
            print("\nValidation For Experiment: ", exp_dir)
            print(time.strftime('%H:%M:%S', time.localtime()))
            val(epoch)


if __name__ == "__main__":
    main()
