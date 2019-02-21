import sys
import os
sys.path.append("../")

import argparse
import random

import torch
import numpy as np

from trainer import Trainer
from attacker import Attacker

import dataloader
from common.logger import Logger
from common.utils import show_current_model
import submodules.models as models
import submodules.attacks as attacks
import submodules.defenses as defenses


logger = Logger("common")

def main(args, scope):

    SEED = args.seed
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    train_loader, val_loader = dataloader.get_loader(args)
    if args.mode == 'train':
        logger.log("Training start!")
        trainer = Trainer(train_loader, val_loader, args)
        if args.verbose <= 10:
            show_current_model(trainer.model, args)
        trainer.train()
        logger.log("Training end!")

    elif args.mode == 'infer':
        logger.log("Evaluation start!")
        trainer = Trainer(None, val_loader, args)
        if args.verbose <= 10:
            show_current_model(trainer.model, args)
        trainer.infer()
        logger.log("Evaluation end!")

    elif args.mode == 'attack':
        logger.log("Experiment start!")
        attacker = Attacker(val_loader, args)
        if args.verbose <= 10:
            show_current_model(attacker.model, args)
        attacker.run()
        logger.log("Experiment end!")

    elif args.mode == 'print_accuracy':
        logger.log("Print accuracy for range of lambdas")
        trainer = Trainer(None, val_loader, args)
        lambdas = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999, 1.0]
        for l in lambdas:
            trainer.model.lambdas = [l]
            logger.log("="*10 + " lambda: {:.3f} ".format(l) + "="*10)
            trainer.infer()
        logger.log("Finished!")

    elif args.mode == 'occlusion':
        logger.log("Print accuracy for range of occlusion ratios")
        args.attack = 'occ_center'
        attacker = Attacker(val_loader, args)
        ratios = np.linspace(0,1,11)
        for r in ratios:
            attacker.attack.ratio = r
            logger.log("="*10 + " ratio: {:.2f} ".format(r) + "="*10)
            attacker.run()
        logger.log("Finished!")

    else:
        raise NotImplementedError


if __name__ == '__main__':
    dataset_names = sorted(name for name in dir(dataloader))
    model_names = sorted(name for name in dir(models))
    attack_names = sorted(name for name in dir(attacks))
    defense_names = sorted(name for name in dir(defenses))

    parser = argparse.ArgumentParser(description='ACE Defense on NSML')
    parser.add_argument("--mode", default=None, type=str)
    parser.add_argument("--seed", default=500, type=int)

    # Log options
    parser.add_argument("--verbose", default=1, type=int,
                        help="verbose level for logger")
    parser.add_argument("--log_step", default=50, type=int)
    parser.add_argument("--log_dir", default='experiments', type=str)

    # Datasets
    parser.add_argument('--data_dir', default='data', type=str,
                        help="location of the dataset directory")
    parser.add_argument('--dataset', default='CIFAR10', type=str, choices=dataset_names,
                        help='available datasets: ' + ' | '.join(dataset_names))
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-b', '--batch_size', default=32, type=int,
                        metavar='N', help='mini-batch size (default: 32)')
    parser.add_argument('--shuffle', action='store_true',
                        help="shuffle the dataset every epoch")

    # Architecture
    parser.add_argument('--model', '-a', metavar='MODEL', default='vgg11', choices=model_names,
                        help='model architecture: ' + ' | '.join(model_names) + ' (default: vgg11)')
    parser.add_argument('--source', default=None, choices=model_names,
                        help='model architecture: ' + ' | '.join(model_names) + ' (default: vgg11)')
 
    # Optimization options
    parser.add_argument("--optimizer", default="SGD", type=str.lower,
                        choices=['sgd', 'adam', 'rmsprop', 'sgd_nn', 'adadelta'])
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')

    # Training options
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--learning_rate', default=0.01, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--weight_decay', '--wd', default=5e-4, type=float,
                        metavar='W', help='weight decay (default: 5e-4)')
    parser.add_argument('--half', dest='half', action='store_true',
                        help='use half-precision(16-bit) ')

    parser.add_argument('--ckpt_name', default=None, type=str,
                        help='name of the checkpoint file')
    parser.add_argument('--ckpt_dir', default='checkpoints', type=str,
                        help='directory used to save the trained models')
    parser.add_argument("--pretrained", action='store_true',
                        help="whether to use a pretrained model." + \
                        "the model must be saved in the checkpoint directory.")
    
    parser.add_argument('--no_eval', action='store_true',
                        help="no evaluation on the validation set")
    parser.add_argument('--recon_ratio', default=0.0, type=float,
                        help="ratio between classification loss and reconstruction loss")

    parser.add_argument('--adv_ratio', default=None, type=float,
                        help="ratio between original loss and adversarial loss")
    parser.add_argument('--distill_ratio', default=0, type=float,
                        help="ratio between the true label and soft label")
    parser.add_argument('--distill_T', default=0, type=float,
                        help="temperature for creating soft labels")
    parser.add_argument('--mixup', default=None, type=float,
                        help="alpha for mixup training")
    
    parser.add_argument('--fine_tune', action='store_true',
                        help="fine-tune the classifier attached to ace module")


    # Attack options
    parser.add_argument('--attack', default=None, type=str, choices=attack_names,
                        help='available algorithms: ' + ' | '.join(attack_names))
    parser.add_argument('--ckpt_src', default=None, type=str,
                        help='Name of the checkpoint file for the source model')

    parser.add_argument('--defense', default=None, type=str, choices=defense_names,
                        help='available algorithms: ' + ' | '.join(defense_names))


    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    args.multigpu = (torch.cuda.device_count() > 1)

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
        args.num_classes = None
        args.no_eval = True

    main(args, scope=locals())
