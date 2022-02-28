# Neural Retargeting

Code for the paper "Kinematic Motion Retargeting via Neural Latent Optimization for Learning Sign Language"

[![arXiv](https://img.shields.io/badge/arXiv-2103.08882-00ff00.svg)](https://arxiv.org/abs/2103.08882)
[![YouTube](https://img.shields.io/badge/YouTube-Video-green.svg)](https://www.youtube.com/watch?v=pX-uie3vLMA)
[![Bilibili](https://img.shields.io/badge/Bilibili-Video-blue.svg)](https://www.bilibili.com/video/BV1mh411Q7BR?share_source=copy_web)

## Prerequisite

- [**PyTorch**](https://pytorch.org/get-started/locally/) Tensors and Dynamic neural networks in Python with strong GPU acceleration
- [**pytorch_geometric**](https://github.com/rusty1s/pytorch_geometric) Geometric Deep Learning Extension Library for PyTorch
- [**Kornia**](https://github.com/kornia/kornia) a differentiable computer vision library for PyTorch.
- [**HDF5 for Python**](https://docs.h5py.org/en/stable/) The h5py package is a Pythonic interface to the HDF5 binary data format.


## Dataset

The Chinese sign language dataset can be downloaded [here](https://www.jianguoyun.com/p/DYm5RzMQ74eHChj_lJ0E).

## Model

The pretrained model can be downloaded [here](https://www.jianguoyun.com/p/DSl6o3EQy96PCBiN750E).

## Get Started

**Training**
```bash
CUDA_VISIBLE_DEVICES=0 python main.py --cfg './configs/train/yumi.yaml'
```

**Inference**
```bash
CUDA_VISIBLE_DEVICES=0 python inference.py --cfg './configs/inference/yumi.yaml'
```

## Simulation Experiment

<img src=https://raw.githubusercontent.com/0aqz0/yumi-gym/master/images/yumi.png width="600">

We build the simulation environment using pybullet, and the code is in this [repository](https://github.com/0aqz0/yumi-gym).

After inference is done, the motion retargeting results are stored in a h5 file. Then run the sample code [here](https://github.com/0aqz0/yumi-gym/tree/master/examples).

## Real-World Experiment

Real-world experiments could be conducted on ABB's YuMi dual-arm collaborative robot equipped with Inspire-Robotics' dexterous hands.

We release the code in this [repository](https://github.com/0aqz0/yumi-control), please follow the instructions.

## Citation

If you find this project useful in your research, please cite this paper.

```
@article{zhang2022kinematic,
  title={Kinematic Motion Retargeting via Neural Latent Optimization for Learning Sign Language},
  author={Zhang, Haodong and Li, Weijie and Liu, Jiangpin and Chen, Zexi and Cui, Yuxiang and Wang, Yue and Xiong, Rong},
  journal={IEEE Robotics and Automation Letters},
  year={2022},
  publisher={IEEE}
}
```
