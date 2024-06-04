# Balanced Coarse-to-Fine Federated Learning For Noisy Heterogeneous Clients

## About
This is a Pytorch implementation of BCFFL. This project is highly borrowed from [Robust Federated Learning with Noisy and Heterogeneous Clients]([https://github.com/SamitHuang/EV_GCN.git](https://github.com/FangXiuwen/Robust_FL)) (CVPR 2022) by Xiuwen Fang, and Mang Ye.

## Prerequisites
- `Python 3.7.4+`
- `Pytorch 1.4.0`
- `torch-geometric `
- `scikit-learn`
- `NumPy 1.16.2`

Ensure Pytorch 1.4.0 is installed before installing torch-geometric. 

This code has been tested using `Pytorch` on a GTX3080TI GPU.

## Dataset
Our experiments are conducted on two datasets, Cifar10 and Cifar100. We set public dataset on the server as a subset of Cifar100, and randomly divide Cifar10 to different clients as private datasets.

Dataset used: [CIFAR-10、CIFAR-100](http://www.cs.toronto.edu/~kriz/cifar.html)

Note: Data will be processed in init_data.py

## Training
To train on ABIDE dataset, please run
```
# init public data and local data
python Dataset/init_data.py
# pretrain local models
python Network/pretrain.py
# BCFFL
python -u BCFFL/BCFFL.py first
```

## Reference 
If you find this code useful in your work, please cite:
