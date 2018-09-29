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
from attacker import Attacker
from defender import Defender
from analysis import Analysis

import submodules.models as models
import submodules.attacks as attacks
import submodules.defenses as defenses
from dataloader import get_loader
from common.utils import find_class_by_name
from common.logger import Logger
import settings
import nsml


logger = Logger("common")

def main(args, scope):

    SEED = args.seed
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    train_loader, val_loader  = get_loader(args.dataset,
            batch_size=args.batch_size,
            num_workers=args.workers
    )
    #model = find_class_by_name(args.model, [nets])(args)

    if args.pause:
        nsml.paused(scope=scope)

    if args.mode == 'train':
        logger.log("Training start!")
        if args.adversarial:
            logger.log("Adversarial Training with {}....".format(args.attack))
            trainer = AdvTrainer(train_loader, val_loader, args)
        elif args.autoencoder:
            trainer = AETrainer(train_loader, val_loader, args)
        else:
            trainer = Trainer(train_loader, val_loader, args)
        trainer.show_current_model()
        save, load, infer = get_binds(trainer)
        nsml.bind(save=save, load=load, infer=infer)

        trainer.train()
        logger.log("Training end!")

    elif args.mode == 'attack':
        logger.log("Attack start!")
        trainer = Attacker(val_loader, args)
        trainer.show_current_model()
        save, load, infer = get_binds(trainer)
        nsml.bind(save=save, load=load, infer=infer)

        trainer.attack()
        logger.log("Attack end!")

    elif args.mode == 'defense':
        logger.log("Defense start!")
        trainer = Defender(val_loader, args)
        trainer.show_current_model()
        save, load, infer = get_binds(trainer)
        nsml.bind(save=save, load=load, infer=infer)

        trainer.defend()
        logger.log("Defend end!")

    elif args.mode == 'analysis':
        logger.log("Analysis start!")
        trainer = Analysis(val_loader, args)
        save, load, infer = get_binds(trainer)
        nsml.bind(save=save, load=load, infer=infer)

        trainer.analysis()
        logger.log("Analysis end!")

    arg_file = os.path.join(str(trainer.log_path), 'args.json')
    with open(arg_file, 'w') as outfile:
        json.dump(vars(args), outfile)


def get_binds(trainer):
    def save(filename=None, *args):
        trainer.save(filename)

    def load(filename=None, *args):
        trainer.load(filename)

    def infer(input):
        # train.infer(input)
        pass

    return save, load, infer

if __name__ == '__main__':
    model_names = sorted(name for name in dir(models))
    attack_names = sorted(name for name in dir(attacks))
    defense_names = sorted(name for name in dir(defenses))

    parser = argparse.ArgumentParser(description='Image classification')

    # Datasets
    parser.add_argument('--dataset', default='CIFAR10', type=str,
                        choices=['MNIST', 'CIFAR10', 'CIFAR100', 'ImageNet', 'TinyImageNet'])
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')

    # Architecture
    parser.add_argument('--model', '-a', metavar='MODEL', default='vgg11', choices=model_names,
                        help='model architecture: ' + ' | '.join(model_names) + ' (default: vgg11)')
    parser.add_argument("--activation", default="relu", type=str)
    parser.add_argument("--beta", default=1, type=float)
    parser.add_argument("--dropout", default=0, type=float)
    parser.add_argument("--conv_weight_init", default='xavier_normal', type=str,
                        help="weight initialization for convolution",
                        choices=dir(torch.nn.init))
    parser.add_argument("--sn", action='store_true', help="Spectral Normalization")

    # Optimization options
    parser.add_argument("--optimizer", default="SGD", type=str.lower,
                        choices=['sgd', 'adam', 'rmsprop', 'sgd_nn', 'adadelta'])
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')

    # Run options
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch_size', default=128, type=int,
                        metavar='N', help='mini-batch size (default: 128)')
    parser.add_argument('--learning_rate', default=0.1, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--weight_decay', '--wd', default=5e-4, type=float,
                        metavar='W', help='weight decay (default: 5e-4)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--half', dest='half', action='store_true',
                        help='use half-precision(16-bit) ')
    parser.add_argument('--save_dir', dest='save_dir',
                        help='The directory used to save the trained models',
                        default='save_temp', type=str)
    parser.add_argument("--pretrained", action='store_true',
                        help="Whether to use a pretrained model." + \
                        "The model must be saved in the checkpoint directory.")
    parser.add_argument("--log_step", default=50, type=int)
    parser.add_argument("--seed", default=500, type=int)
    parser.add_argument("--multigpu", default=0, type=int,
                        help="Number of gpus to use. Batchsize // ngpu = 0")
    parser.add_argument("--record", dest='record', action='store_true')
    parser.add_argument("--no_cuda", action='store_true', help="For CPU inference")
    parser.add_argument("--ckpt_name", type=str)

    # Adaptive ReLU settings
    parser.add_argument("--lam", default=1, type=float,
                        help="constant for the Rademacher complexity constraint")
    parser.add_argument("--schedule_id", default=1, type=int,
                        help="Adaptive update schedule scheme, look trainer.py for now")
    parser.add_argument("--no_affine", action='store_true', help="Deactivate affine BN transformation")
    parser.add_argument("--no_bn", action='store_true', help="Deactivate BN transformation")
    parser.add_argument("--normalization", default="batch", type=str,
                        help="type of normalization layer")

    # Adversarial attack settings
    parser.add_argument('--attack', metavar='ATTACK', default=None, choices=attack_names,
                        help='attack algorithm: ' + ' | '.join(attack_names) + ' (default: None)')
    parser.add_argument('--aggregate', '--ag', default='accdrop', choices=['success', 'accdrop'],
                        help='the criterion for ensembling several attack algorithms')
    parser.add_argument('--avg_step', default=1, type=int)
    parser.add_argument('--img_log_step', default=20, type=int)
    parser.add_argument('--log_artifact', action='store_true',
                        help="calculate the ratio of perturbation on the artifacts of the network")
    parser.add_argument('--target', default=None, type=int,
                        help="None for non-targeted attacks, -1 for least likely attack")
    parser.add_argument('--T', default=None, type=float)
    parser.add_argument("--domain_restrict", action='store_true')

    # JSMA
    parser.add_argument("--top_overlap", default=0, type=int, help="Overlap search domain to select")
    parser.add_argument("--gamma", default=0.1, type=float, help="% of pixels to use in image")
    # DeepFool
    parser.add_argument("--max_iter", default=50, type=int, help="Max iteration to perturb")

    # nsml setting
    parser.add_argument("--mode", default='train', type=str,
                        choices=['train', 'attack', 'defense', 'analysis'])
    parser.add_argument("--pause", type=int, default=0)
    parser.add_argument("--iteration", type=str, default='0')

    # Log options
    parser.add_argument("--verbose", default=1, type=int,
                        help="verbose level for logger")

    # Defense settings
    parser.add_argument("--defense", type=str, default=None, choices=defense_names,
                        help='defense algorithm: ' + ' | '.join(defense_names) + ' (default: None)')
    parser.add_argument("--source", type=str, default=None, choices=model_names)
    parser.add_argument("--ckpt_src", type=str)

    # Enhancer (ACE)
    parser.add_argument("--lambd", type=float, default=0.1)
    parser.add_argument("--fine_tune", action='store_true', help="fine-tune the enhancer module")

    # PixelShift defense
    parser.add_argument("--pad_type", type=str, default="replication",
                        choices=["reflection", "replication", "zero"])
    parser.add_argument("--x_coord", type=int, default=1)
    parser.add_argument("--y_coord", type=int, default=0)
    parser.add_argument("--random", action='store_true')

    # Pixel Deflection defense
    parser.add_argument("--ndeflection", type=int, default=200)
    parser.add_argument("--window", type=int, default=10)
    parser.add_argument("--sigma", type=float, default=0.04)

    # Boundary analysis options
    parser.add_argument("--max_eps", default=100, type=float,
                        help="the maximum size of perturbation")
    parser.add_argument("--len_eps", default=200, type=int,
                        help="number of sample perturbations")
    parser.add_argument("--img_first", default=0, type=int)
    parser.add_argument("--img_last", default=99, type=int)

    # EOT
    parser.add_argument("--eot_attack", default='pgd', type=str)
    parser.add_argument("--eot_norm", default='l2', type=str)
    parser.add_argument("--nsamples", default=30, type=int)
    parser.add_argument("--eot_iter", default=1000, type=int)

    # Adversarial Training
    parser.add_argument("--adversarial", action='store_true', help="Flag for adversarial training")
    parser.add_argument("--alpha", default=0.5, type=float,
                        help="Ratio between adversarial loss and clean loss")

    # Autoencoder Options
    parser.add_argument("--autoencoder", action='store_true', help="Flag for autoencoder training")
    parser.add_argument("--ckpt_ae", type=str, default=None)


    args = parser.parse_args()
    args.cuda = False if args.no_cuda else True

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

    # Default options for analysis mode
    if args.mode == 'analysis':
        args.batch_size = 1
        args.learning_rate = 0.001
        args.optimizer = 'Adam'

    if args.T is not None:
        args.domain_restrict = True

    #import multiprocessing
    #print(multiprocessing.cpu_count())
    #import psutil
    #print(psutil.cpu_percent())
    #print(psutil.virtual_memory())
    main(args, scope=locals())
