---
layout: post
title: Summary of Deep Neural Network Optimization on Resource-Constraint Devices
---

## Prune
*  Learning both Weights and Connections for Efficient Neural Networks
Classical

* Dynamic Network Surgery for Efficient DNNs
Classical


## Structure Prunning
* Pruning Filters for Efficient ConvNets
看filter绝对值去
## Quantization (Binarization, Ternarization)
* Towards the Limit of Network Quantization
Clustering + Approximate Hessian

* Trained Ternary Quantization

* DoReFa-Net

* Training and Inference with Integers in Deep Neural Network (WAGE)

* XNOR-Net

* Binarized Neural Networks: Training Neural Networks with Weights and Activations Constrained to +1 or -1
Classical

* BinaryConnect: Training Deep Neural Networks with Binary Weights during Propagations
Classical

* Loss-Aware Binaration of Deep Networks

* Ternary Wight Networks

* Incremental Network Quantization

* Extremely Low Bit Neural Network: Squeeze the Last Bit out with Admm

## Optimization from Software
* Faster CNNs with Direct Sparse Convolutions and Guided Prunning
从内存角度出发，通过底层操作加速卷积操作，且有对useful sparsity的推导和证明，对比实验很充分。
* SBNet: Sparse Blocks Networks for Fast Inference
思想很简单，把非稀疏的input聚在一起处理。因为Dense Convolution已经被优化得很好，但工程性极强。

##  Miscellaneous
* Compressing Neural Network with the Hashing Trick

* Interleaved Group Convolutions for Deep Neural Networks
按channel切开input feature做卷积（早期AlexNet分组卷积思想），再把来自不同的partition的feature maps组一起做第二次卷积