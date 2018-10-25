# ACE: Adversarial Checkerboard Enhancer to Induce and Evade Adversarial Attacks

This repository contains a PyTorch implementation for the paper
ACE: Adversarial Checkerboard Enhancer to Induce and Evade Adversarial Attacks (arXiv).


## Usage

This repository contains tools for both training computer vision models from scratch, and conducting experiments on adversarial attack/defense. The specific task to be performed can be chosen by the argument `--mode`. Currently there are 4 options: `train`, `train_ae`, `train_adv`, `defense`.

Here are some common options.

- `--multigpu`: Number of gpus to use. Defaults to 1.

- `--dataset`: Dataset to train or experiment on. Available choices are 'MNIST', 'CIFAR10', 'CIFAR100', 'ImageNet', and 'TinyImageNet'. MNIST and CIFAR can be downloaded online from the Pytorch dataloader. ImageNet and TinyImageNet can also be used but should be prepared beforehand.

- `--batch_size`: The batch size of image samples. Defaults to 128.

- `--verbose`: Verbose level of our logger. Messages with level lower than this argument are suppressed. Messages for debugging are set to level 10. Defaults to 1.

#### train

This is the mode for training model architectures for computer vision, e.g. ResNet and VGG.

- `--model`: The model architecture. Currently available choices are LeNet_toy, ResNet, and VGG models. If `--dataset` is set to ImageNet, models from `torchvision.models` are also available.

- `--optimizer`: Optimizer for gradient descent. Available choices are 'sgd', 'adam', 'rmsprop', 'sgd_nn', and 'adadelta'. 'sgd_nn' stands for SGD optimizer without Nesterov momentum.

- `--learning_rate`: Learning_rate for gradient descent.


#### train_ae

#### train_adv

#### defense


## Paper Preview

## Citation

```
@article{ACE,
    title={ACE: Adversarial Checkerboard Enhancer to Induce and Evade Adversarial Attacks},
    author={Hwang, Jisung and Kim, Younghoon and Chun, Sanghyuk and Yoo, Jaejun and Kim, Ji-Hoon and Han, Dongyoon and Ha, Jung-Woo},
    journal={arXiv},
    year={2018}
}
```

## Contact

Jisung Hwang (), Younghoon Kim (yh01dlx@gmail.com)
