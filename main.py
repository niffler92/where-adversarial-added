#-*- coding: utf-8 -*-
from __future__ import print_function, unicode_literals, absolute_import, division
import sys
import os
sys.path.append("../")
import argparse
from datetime import datetime
import random
import json

import torch
torch.backends.cudnn.enabled = True
import torchvision.transforms as transforms
import numpy as np

from trainer import Trainer, AdvTrainer, AETrainer
from defender import Defender

import submodules.models as models
import submodules.attacks as attacks
import submodules.defenses as defenses
from dataloader import get_loader
from common.logger import Logger
import settings


logger = Logger("common")

def main(args, scope):

    SEED = args.seed
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    train_loader, val_loader = get_loader(args.dataset,
            batch_size=args.batch_size,
            num_workers=args.workers
    )

    if args.mode == 'train':
        logger.log("Training start!")
        args.autoencoder = False

        runner = Trainer(train_loader, val_loader, args)
        runner.show_current_model()
        runner.train()
        logger.log("Training end!")

    elif args.mode == 'train_adv':
        logger.log("Adversarial Training with {}....".format(args.attack))
        args.autoencoder = False

        runner = AdvTrainer(train_loader, val_loader, args)
        runner.show_current_model()
        runner.train()
        logger.log("Training end!")

    elif args.mode == 'train_ae':
        logger.log("Training start!")
        args.autoencoder = True
        args.ckpt_name = args.ckpt_ae

        runner = AETrainer(train_loader, val_loader, args)
        runner.show_current_model()
        runner.train()
        logger.log("Training end!")

    elif args.mode == 'defense':
        logger.log("Defense start!")
        args.autoencoder = False

        runner = Defender(val_loader, args)
        runner.show_current_model()
        runner.defend()
        logger.log("Defense end!")

    arg_file = os.path.join(str(runner.log_path), 'args.json')
    with open(arg_file, 'w') as outfile:
        json.dump(vars(args), outfile)


if __name__ == '__main__':
    model_names = sorted(name for name in dir(models))
    attack_names = sorted(name for name in dir(attacks))
    defense_names = sorted(name for name in dir(defenses))

    parser = argparse.ArgumentParser(description='ACE-Defense')
    parser.add_argument('--mode', default='train', type=str,
                        choices=['train', 'train_adv', 'train_ae', 'defense'])

    # Common options
    parser.add_argument("--seed", default=500, type=int)
    parser.add_argument("--multigpu", default=0, type=int,
                        help="Number of gpus to use. Batchsize // ngpu = 0")
    parser.add_argument("--no_cuda", action='store_true', help="For CPU inference")
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--half', dest='half', action='store_true',
                        help='use half-precision(16-bit) ')

    parser.add_argument('--dataset', default='CIFAR10', type=str,
                        choices=['MNIST', 'CIFAR10', 'CIFAR100', 'ImageNet', 'TinyImageNet'])
    parser.add_argument('-b', '--batch_size', default=128, type=int,
                        metavar='N', help='mini-batch size (default: 128)')
    parser.add_argument("--verbose", default=1, type=int,
                        help="verbose level for logger")
    parser.add_argument("--log_step", default=50, type=int)
    parser.add_argument('--img_log_step', default=20, type=int)

    # Training options
    parser.add_argument('--model', '-a', metavar='MODEL', default='vgg11', choices=model_names,
                        help='model architecture: ' + ' | '.join(model_names) + ' (default: vgg11)')
    parser.add_argument("--conv_weight_init", default='xavier_normal', type=str,
                        help="weight initialization for convolution",
                        choices=dir(torch.nn.init))
    parser.add_argument("--pretrained", action='store_true',
                        help="Whether to use a pretrained model." + \
                        "The model must be saved in the checkpoint directory.")
    parser.add_argument("--ckpt_name", type=str, default=None)

    parser.add_argument("--optimizer", default="sgd", type=lambda s: s.encode('utf8').lower(),
                        choices=['sgd', 'adam', 'rmsprop', 'sgd_nn', 'adadelta'])
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--learning_rate', default=0.01, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--weight_decay', '--wd', default=5e-4, type=float,
                        metavar='W', help='weight decay (default: 5e-4)')

    parser.add_argument("--alpha", default=0.5, type=float,
                        help="Ratio between adversarial loss and clean loss")
    parser.add_argument("--ckpt_ae", type=str, default=None)

    # Adversarial attack/defense options
    parser.add_argument("--source", type=str, default=None, choices=model_names)
    parser.add_argument("--ckpt_src", type=str)

    parser.add_argument('--attack', metavar='ATTACK', default=None, choices=attack_names,
                        help='attack algorithm: ' + ' | '.join(attack_names) + ' (default: None)')
    parser.add_argument('--target', default=None, type=int,
                        help="None for non-targeted attacks, -1 for least likely attack")
    parser.add_argument('--G', default=None, type=float)
    parser.add_argument("--gamma", default=0.1, type=float, help="% of pixels to use in image")
    parser.add_argument("--max_iter", default=50, type=int, help="Max iteration to perturb")
    parser.add_argument("--eot_attack", default='pgd', type=str)
    parser.add_argument("--eot_norm", default='l2', type=str)
    parser.add_argument("--nsamples", default=30, type=int)
    parser.add_argument("--eot_iter", default=1000, type=int)

    parser.add_argument("--defense", type=str, default=None, choices=defense_names,
                        help='defense algorithm: ' + ' | '.join(defense_names) + ' (default: None)')
    parser.add_argument("--lambd", type=float, default=0)
    parser.add_argument("--pad_type", type=str, default="replication",
                        choices=["reflection", "replication", "zero"])
    parser.add_argument("--x_coord", type=int, default=1)
    parser.add_argument("--y_coord", type=int, default=0)
    parser.add_argument("--random", action='store_true')
    parser.add_argument("--ndeflection", type=int, default=200)
    parser.add_argument("--window", type=int, default=10)
    parser.add_argument("--sigma", type=float, default=0.04)


    args = parser.parse_args()
    args.cuda = False if args.no_cuda else torch.cuda.is_available()

    # Set num_classes according to dataset
    if args.dataset in ['MNIST', 'CIFAR10']:
        args.num_classes = 10
    elif args.dataset == "CIFAR100":
        args.num_classes = 100
    elif args.dataset == "TinyImageNet":
        args.num_classes = 200
    elif args.dataset == "ImageNet":
        args.num_classes = 1000
    else:
        raise NotImplementedError

    if args.G is not None:
        args.domain_restrict = True
    else:
        args.domain_restrict = False

    main(args, scope=locals())
