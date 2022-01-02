# PyTorch implementation of 'Denoising Diffusion Probabilistic Models'

This repository contains my attempt at reimplementing the main algorithm and model presenting in `Denoising Diffusion Probabilistic Models`, the recent paper by [Ho et al., 2020](https://arxiv.org/abs/2006.11239). A nice summary of the paper by the authors is available [here](https://hojonathanho.github.io/diffusion/). 

This implementation uses pytorch lightning to limit the boilerplate as much as possible. Due to time and computational constraints, I only experimented with 32x32 image datasets, but it should scale up to larger datasets like LSUN and CelebA as demonstrated in the original paper. This implementation was done for my own self-education, and hopefully it can help others learn as well.

Use the provided [`entry.ipynb`](./entry.ipynb) notebook to train model and sample generated images. 

Supports MNIST, Fashion-MNIST and CIFAR datasets.

## Requirements

* PyTorch
* PyTorch-Lightning
* Torchvision
* imageio (for gif generation)

## Generated Images

### MNIST

![MNIST Generation](/imgs/mnist.gif)

### Fashion-MNIST

![Fashion MNIST Generation](/imgs/fashion.gif)

### CIFAR

![CIFAR Generation](/imgs/cifar.gif)
