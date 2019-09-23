import sys
import os
abs_file_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(abs_file_path, "..", "..")) # add path

import torch
import torch.nn as nn
import collections
from torch.utils.serialization import load_lua
from model import SCNN

model1 = load_lua('experiments/vgg_SCNN_DULR_w9/vgg_SCNN_DULR_w9.t7', unknown_classes=True)
model2 = collections.OrderedDict()

model2['backbone.0.weight'] = model1.modules[0].weight
model2['backbone.1.weight'] = model1.modules[1].weight
model2['backbone.1.bias'] = model1.modules[1].bias
model2['backbone.1.running_mean'] = model1.modules[1].running_mean
model2['backbone.1.running_var'] = model1.modules[1].running_var
model2['backbone.3.weight'] = model1.modules[3].weight
model2['backbone.4.weight'] = model1.modules[4].weight
model2['backbone.4.bias'] = model1.modules[4].bias
model2['backbone.4.running_mean'] = model1.modules[4].running_mean
model2['backbone.4.running_var'] = model1.modules[4].running_var

model2['backbone.7.weight'] = model1.modules[7].weight
model2['backbone.8.weight'] = model1.modules[8].weight
model2['backbone.8.bias'] = model1.modules[8].bias
model2['backbone.8.running_mean'] = model1.modules[8].running_mean
model2['backbone.8.running_var'] = model1.modules[8].running_var
model2['backbone.10.weight'] = model1.modules[10].weight
model2['backbone.11.weight'] = model1.modules[11].weight
model2['backbone.11.bias'] = model1.modules[11].bias
model2['backbone.11.running_mean'] = model1.modules[11].running_mean
model2['backbone.11.running_var'] = model1.modules[11].running_var

model2['backbone.14.weight'] = model1.modules[14].weight
model2['backbone.15.weight'] = model1.modules[15].weight
model2['backbone.15.bias'] = model1.modules[15].bias
model2['backbone.15.running_mean'] = model1.modules[15].running_mean
model2['backbone.15.running_var'] = model1.modules[15].running_var
model2['backbone.17.weight'] = model1.modules[17].weight
model2['backbone.18.weight'] = model1.modules[18].weight
model2['backbone.18.bias'] = model1.modules[18].bias
model2['backbone.18.running_mean'] = model1.modules[18].running_mean
model2['backbone.18.running_var'] = model1.modules[18].running_var
model2['backbone.20.weight'] = model1.modules[20].weight
model2['backbone.21.weight'] = model1.modules[21].weight
model2['backbone.21.bias'] = model1.modules[21].bias
model2['backbone.21.running_mean'] = model1.modules[21].running_mean
model2['backbone.21.running_var'] = model1.modules[21].running_var

model2['backbone.24.weight'] = model1.modules[24].weight
model2['backbone.25.weight'] = model1.modules[25].weight
model2['backbone.25.bias'] = model1.modules[25].bias
model2['backbone.25.running_mean'] = model1.modules[25].running_mean
model2['backbone.25.running_var'] = model1.modules[25].running_var
model2['backbone.27.weight'] = model1.modules[27].weight
model2['backbone.28.weight'] = model1.modules[28].weight
model2['backbone.28.bias'] = model1.modules[28].bias
model2['backbone.28.running_mean'] = model1.modules[28].running_mean
model2['backbone.28.running_var'] = model1.modules[28].running_var
model2['backbone.30.weight'] = model1.modules[30].weight
model2['backbone.31.weight'] = model1.modules[31].weight
model2['backbone.31.bias'] = model1.modules[31].bias
model2['backbone.31.running_mean'] = model1.modules[31].running_mean
model2['backbone.31.running_var'] = model1.modules[31].running_var

model2['backbone.34.weight'] = model1.modules[33].weight
model2['backbone.35.weight'] = model1.modules[34].weight
model2['backbone.35.bias'] = model1.modules[34].bias
model2['backbone.35.running_mean'] = model1.modules[34].running_mean
model2['backbone.35.running_var'] = model1.modules[34].running_var
model2['backbone.37.weight'] = model1.modules[36].weight
model2['backbone.38.weight'] = model1.modules[37].weight
model2['backbone.38.bias'] = model1.modules[37].bias
model2['backbone.38.running_mean'] = model1.modules[37].running_mean
model2['backbone.38.running_var'] = model1.modules[37].running_var
model2['backbone.40.weight'] = model1.modules[39].weight
model2['backbone.41.weight'] = model1.modules[40].weight
model2['backbone.41.bias'] = model1.modules[40].bias
model2['backbone.41.running_mean'] = model1.modules[40].running_mean
model2['backbone.41.running_var'] = model1.modules[40].running_var

model2['layer1.0.weight'] = model1.modules[42].modules[0].weight
model2['layer1.1.weight'] = model1.modules[42].modules[1].weight
model2['layer1.1.bias'] = model1.modules[42].modules[1].bias
model2['layer1.1.running_mean'] = model1.modules[42].modules[1].running_mean
model2['layer1.1.running_var'] = model1.modules[42].modules[1].running_var
model2['layer1.3.weight'] = model1.modules[42].modules[3].weight
model2['layer1.4.weight'] = model1.modules[42].modules[4].weight
model2['layer1.4.bias'] = model1.modules[42].modules[4].bias
model2['layer1.4.running_mean'] = model1.modules[42].modules[4].running_mean
model2['layer1.4.running_var'] = model1.modules[42].modules[4].running_var

model2['message_passing.up_down.weight'] = model1.modules[42].modules[6].modules[0].modules[0].modules[2].modules[0].modules[1].modules[1].modules[0].weight
model2['message_passing.down_up.weight'] = model1.modules[42].modules[6].modules[0].modules[0].modules[140].modules[1].modules[2].modules[0].modules[0].weight
model2['message_passing.left_right.weight'] = model1.modules[42].modules[6].modules[1].modules[0].modules[2].modules[0].modules[1].modules[1].modules[0].weight
model2['message_passing.right_left.weight'] = model1.modules[42].modules[6].modules[1].modules[0].modules[396].modules[1].modules[2].modules[0].modules[0].weight

model2['layer2.1.weight'] = model1.modules[42].modules[8].weight
model2['layer2.1.bias'] = model1.modules[42].modules[8].bias
model2['fc.0.weight'] = model1.modules[43].modules[1].modules[3].weight
model2['fc.0.bias'] = model1.modules[43].modules[1].modules[3].bias
model2['fc.2.weight'] = model1.modules[43].modules[1].modules[5].weight
model2['fc.2.bias'] = model1.modules[43].modules[1].modules[5].bias

save_name = os.path.join('experiments', 'vgg_SCNN_DULR_w9', 'vgg_SCNN_DULR_w9.pth')
torch.save(model2, save_name)

# load and save again
net = SCNN(input_size=(800, 288), pretrained=False)
d = torch.load(save_name)
net.load_state_dict(d, strict=False)
for m in net.backbone.modules():
    if isinstance(m, nn.Conv2d):
        if m.bias is not None:
            m.bias.data.zero_()


save_dict = {
    "epoch": 0,
    "net": net.state_dict(),
     "optim": None,
    "lr_scheduler": None
}

if not os.path.exists(os.path.join('experiments', 'vgg_SCNN_DULR_w9')):
    os.makedirs(os.path.join('experiments', 'vgg_SCNN_DULR_w9'), exist_ok=True)
torch.save(save_dict, save_name)