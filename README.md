# Where to be Adversarial Perturbations Added? Investigating and Manipulating Pixel Robustness Using Input Gradients

This repository contains a PyTorch implementation for the paper
Where to be Adversarial Perturbations Added? Investigating and Manipulating Pixel Robustness Using Input Gradients

## Paper Preview


## Usage

This repository contains tools for both training computer vision models from scratch, and conducting experiments on adversarial attack/defense. The specific task to be performed can be chosen by the argument `--mode`. Currently there are 4 options: `train`, `train_adv`, `train_ae`, `defense`.

Our custom logger saves the stream file and the input arguments in SAVE_DIR, as specified in `settings.py`. Checkpoints of the trained models, output images and arrays are saved in the same directory. You can load pretrained models from checkpoints from LOAD_DIR, which is also specified in `settings.py`.

Here are some common options.

- `--seed`: Random seed for experiments.

- `--multigpu`: Number of gpus to use. Defaults to 1.

- `--dataset`: Dataset to train or experiment on. Available choices are 'MNIST', 'CIFAR10', 'CIFAR100', 'ImageNet', and 'TinyImageNet'. MNIST and CIFAR can be downloaded online from the Pytorch dataloader. ImageNet and TinyImageNet can also be used but should be prepared beforehand.

- `--batch_size`: The batch size of image samples. Defaults to 128.

- `--verbose`: Verbose level of our logger. Messages with level lower than this argument are suppressed. Messages for debugging are set to level 10. Defaults to 1.

- `--log_step`: The frequency for the logger to print out information.


### train

This is the mode for training model architectures for computer vision, e.g. ResNet and VGG.

- `--model`: The model architecture that is defined in `submodules.models`. Currently available choices are LeNet_toy, ResNet, and VGG models. If the dataset is ImageNet, models from `torchvision.models` are also available.

- `--pretrained`: Load pretrained checkpoints from LOAD_DIR or `torchvision.models`.

- `--ckpt_name`: Name of the checkpoint file. If not specified, the checkpoint is assumed to have the same name as the name of the architecture (e.g. ResNet18), under the directory of the same name as the dataset (e.g. CIFAR10).

- `--ckpt_ae`: Name of the checkpoint file for the ACE module. This argument is called only when training networks that contain the ACE module, e.g. 'unet_resnet152'. Such models are specified in `submodules.enhancer`.

- `--lambd`: The intensity parameter λ for the ACE module. Defaults to 0.

- `--optimizer`: Optimizer for gradient descent. Available choices are 'sgd', 'adam', 'rmsprop', 'sgd_nn', and 'adadelta'. 'sgd_nn' stands for SGD optimizer without Nesterov momentum.

- `--learning_rate`: Learning rate for gradient descent.

- `--epochs`: Number of epochs for training.

To call a pretrained ResNet architecture with 18 layers for CIFAR10:
```
python main.py --mode train --dataset CIFAR10 --model ResNet18 --pretrained --ckpt_name CHECKPOINT
```

To call a pretrained VGG architecture from `torchvision.models` combined with a UNet-shaped ACE module with λ=0.9:
```
python main.py --mode train --dataset ImageNet --model unet_vgg19 --pretrained --ckpt_ae CHECKPOINT --lambd 0.9
```

### train_adv

This is the mode for the adversarial training of networks. All arguments are shared with the mode 'train' except for the following.

- `--attack`: The adversarial attack algorithm defined in `submodules.attacks`. Currently available choices are Carlini&Wagner, DeepFool, FGSM, PGD, JSMA, and OnePixel attacks.  

- `--alpha`: The ratio between the adversarial attack loss and the original loss function. Defaults to 0.5.

To train a ResNet model with PGD attacks:
```
python main.py --mode train_adv --dataset CIFAR10 --model ResNet18 --attack pgd
```

### train_ae

This is the mode for training ACE modules separate from the classifier networks.

- `--model`: The autoencoder architecture for ACE defined in `submodules.autoencoder`. Currently available choices are SegNet and UNet-shaped architectures.

- `--pretrained`: Load pretrained checkpoints from LOAD_DIR.

- `--ckpt_name`, `--ckpt_ae`: Name of the checkpoint file. Both arguments can be used for training the ACE module.

To train a SegNet-shaped ACE module:
```
python main.py --mode train_ae --dataset CIFAR10 --model segnet
```

### defense

This is the mode for conducting experiments on adversarial attack and defense algorithms.

- `--attack`: The adversarial attack algorithm defined in `submodules.attacks`. Expectation Over Transformation (EOT) attacks can be performed by providing 'eot' as the input for this argument.

- `--eot_attack`: The base attack algorithm for EOT attacks.

- `--source`: The source model to generate attacks from in a transfer attack scenario. The model must also be defined in `submodules.models`.

- `--ckpt_src`: Name of the checkpoint file for the source model.

- `--defense`: The defense algorithm defined in `submodules.defenses`. Currently available choices are PixelDeflection, Randomization, Region-Based and PixelShift methods.

To perform transfer attacks from a vanilla ResNet to ResNet with the ACE module, and defend them by shifting the image by a single pixel:
```
python main.py --mode defense --dataset CIFAR10 --model unet_resnet152 --lambd 0.9 --pretrained --ckpt_name CHECKPOINT_RESNET --ckpt_ae CHECKPOINT_UNET --attack fgsm --source ResNet152 --ckpt_src CHECKPOINT_RESNET --defense pixelshift
```


## Citation

```
@article{Where,
    title={Where to be adversarial perturbations added? Investigating and manipulating pixel robustness using input gradients},
    booktitle = {7th International Conference on Learning Representations, Debugging Machine Learning Models Workshop, {ICLR} 2019, New Orleans, LA, USA, May 6-9, 2019}
    author={Hwang, Jisung and Kim, Younghoon and Chun, Sanghyuk and Yoo, Jaejun and Kim, Ji-Hoon and Han, Dongyoon and Ha, Jung-Woo},
    year={2019}
}
```

## Contact

Jisung Hwang (jeshwang92@uchicago.edu), Younghoon Kim (yh01dlx@gmail.com)
