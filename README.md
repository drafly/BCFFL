# Balanced Coarse-to-Fine Federated Learning For Noisy Heterogeneous Clients

## About
This is a Pytorch implementation of BCFFL. This project is highly borrowed from [Robust Federated Learning with Noisy and Heterogeneous Clients](https://github.com/FangXiuwen/Robust_FL) (CVPR 2022) by Xiuwen Fang, and Mang Ye.

## Prerequisites
- `Python 3.8.0`
- `Pytorch 1.10.1`
- `NumPy 1.20.1`
- `tensorboardX 2.6.2.2`


This code has been tested using `Pytorch` on a GTX3080TI GPU.

## Dataset
Our experiments are conducted on two datasets, Cifar10 and Cifar100. We set public dataset on the server as a subset of Cifar100, and randomly divide Cifar10 to different clients as private datasets.

Dataset used: [CIFAR-10、CIFAR-100](http://www.cs.toronto.edu/~kriz/cifar.html)

Note: Data will be processed in init_data.py

## Training and Testing
To train on Cifar10 and Cifar100 dataset, please run
```
# init public data and local data
python Dataset/init_data.py

# pretrain local models
python Network/pretrain.py

# BCFFL
python -u BCFFL/BCFFL.py self-space
```

## Reference 
If you find this code useful in your work, please cite:
@article{han2025balanced,
  title={Balanced coarse-to-fine federated learning for noisy heterogeneous clients},
  author={Han, Longfei and Zhai, Ying and Jia, Yanan and Cai, Qiang and Li, Haisheng and Huang, Xiankai},
  journal={Complex \& Intelligent Systems},
  volume={11},
  number={2},
  pages={126},
  year={2025},
  publisher={Springer}
}
