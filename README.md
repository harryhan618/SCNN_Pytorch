# SCNN lane detection in Pytorch

SCNN is a segmentation-tasked lane detection algorithm, described in ['Spatial As Deep: Spatial CNN for Traffic Scene Understanding'](https://arxiv.org/abs/1712.06080). The [official implementation](<https://github.com/XingangPan/SCNN>) is in lua torch.

This repository contains a re-implementation in Pytorch.



## Data preparation

### CULane

The dataset is available in [CULane](<https://xingangpan.github.io/projects/CULane.html>). Please download and unzip the files in one folder, which later is represented as `CULane_path`. 

```
CULane_path
├── driver_100_30frame
├── driver_161_90frame
├── driver_182_30frame
├── driver_193_90frame
├── driver_23_30frame
├── driver_37_30frame
├── laneseg_label_w16
├── laneseg_label_w16_test
└── list
```

Then modify the path of `CULane_path` in `config.py`. **Note: absolute path is encouraged.**



## Demo Test

For single image demo test:

```
python demo_test.py [--visualize] -i demo.jpg -w experiments/exp0/exp0.pth
```

![](demo/demo_result.jpg "demo_result")





## Train 

1. Specify an experiment directory, e.g. `experiments/exp0`.  Assign the path to variable `exp_dir` in `train.py`.

2. Modify the hyperparameters in `cfg.json`.

3. Start training:

   ```python
   python train.py [-r]
   ```

4. Monitor on tensorboard:

   ```
   tensorboard --logdir='experiments/exp0' > experiments/exp0/board.txt 2>&1 &
   ```




- My model is trained with `torch.nn.DataParallel`. Modify it according to your hardware configuration.
- Currently the backbone is vgg16 from torchvision.





## Acknowledgement

This repos is build based on [official implementation](<https://github.com/XingangPan/SCNN>).

