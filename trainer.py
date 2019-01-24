from argparse import Namespace
import time
import os
import torch
import numpy as np

import submodules.ace as ace
import submodules.attacks as attacks

from common.logger import Logger
from common.summary import EvaluationMetrics
from common.torch_utils import get_model, get_optimizer
from common.utils import get_dirname, show_current_model


class Trainer:
    def __init__(self, train_loader, val_loader, args):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.args = Namespace(**vars(args))

        self.model, self.compute_loss = get_model(args)
        # If adversarial training
        if self.args.adv_ratio != 0:
            if self.args.source is not None:
                args.model = self.args.source
                args.ckpt_name = self.args.ckpt_src
                source, _ = get_model(args)
            else:
                source = self.model
            self.scheme = getattr(attacks, self.args.attack)(source, args)

        self.epochs = self.args.epochs
        self.total_step = len(train_loader) * self.args.epochs
        self.step = 0
        self.epoch = 0
        self.start_epoch = 1
        self.lr = self.args.learning_rate
        self.best_loss = np.inf

        self.log_path = get_dirname(self.args)
        self.logger = Logger("train", self.log_path, self.args.verbose)
        self.logger.log("Logs will be saved in {}".format(self.log_path))

        self.logger.add_level('STEP', 21, 'green')
        self.logger.add_level('EPOCH', 22, 'cyan')
        self.logger.add_level('EVAL', 23, 'yellow')

        params = self.model.parameters()
        self.optimizer = get_optimizer(self.args.optimizer, params, self.args)

    def train(self):
        show_current_model(self.model, self.args)
        # DEBUG:
        self.save()
        self.eval()
        for self.epoch in range(self.start_epoch, self.args.epochs+1):
            self.adjust_learning_rate([int(self.args.epochs/2), int(self.args.epochs*3/4)], factor=0.1)
            self.train_epoch()
            self.eval()

    def train_epoch(self):
        self.model.train()
        eval_metrics = EvaluationMetrics(['Loss', 'Acc', 'Time'])

        for i, (images, labels) in enumerate(self.train_loader):
            st = time.time()
            self.step += 1
            if self.args.cuda:
                images = images.cuda()
                labels = labels.cuda()
            if self.args.half:
                images = images.half()

            outputs, loss = self.compute_loss(self.model, images, labels)
            # If adversarial training
            if self.args.adv_ratio != 0:
                adv_images = self.scheme.generate(images, labels)
                adv_outputs, adv_loss = self.compute_loss(self.model, adv_images, labels)
                
                outputs = torch.cat([outputs, adv_outputs])
                labels = torch.cat([labels, labels])
                loss = self.args.adv_ratio*adv_loss + (1 - self.args.adv_ratio)*loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            elapsed_time = time.time() - st
            loss = loss.item()
            if outputs is not None:
                _, preds = torch.max(outputs.float(), 1)
                accuracy = (labels == preds).float().mean().item()
            else:
                accuracy = np.nan

            batch_size = labels.size(0)
            eval_metrics.update('Loss', loss, batch_size)
            eval_metrics.update('Acc', accuracy, batch_size)
            eval_metrics.update('Time', elapsed_time, batch_size)

            if self.step % self.args.log_step == 0:
                self.logger.scalar_summary(eval_metrics.val, self.step, 'STEP')

        self.logger.scalar_summary(eval_metrics.avg, self.step, 'EPOCH')

    def eval(self):
        self.model.eval()
        eval_metrics = EvaluationMetrics(['Loss', 'Acc', 'Time'])

        for i, (images, labels) in enumerate(self.val_loader):
            st = time.time()
            if self.args.cuda:
                images = images.cuda()
                labels = labels.cuda()
            if self.args.half:
                images = images.half()

            outputs, loss = self.compute_loss(self.model, images, labels)
            # If adversarial training
            if self.args.adv_ratio != 0:
                adv_images = self.scheme.generate(images, labels)
                adv_outputs, adv_loss = self.compute_loss(self.model, adv_images, labels)
                
                outputs = torch.cat([outputs, adv_outputs])
                labels = torch.cat([labels, labels])
                loss = self.args.adv_ratio*adv_loss + (1 - self.args.adv_ratio)*loss

            elapsed_time = time.time() - st
            loss = loss.item()
            if outputs is not None:
                _, preds = torch.max(outputs.float(), 1)
                accuracy = (labels == preds).float().mean().item()
            else:
                accuracy = np.nan

            batch_size = labels.size(0)
            eval_metrics.update('Loss', loss, batch_size)
            eval_metrics.update('Acc', accuracy, batch_size)
            eval_metrics.update('Time', elapsed_time, batch_size)

        # Save best model
        if eval_metrics.avg['Loss'] < self.best_loss:
            self.save()
            self.logger.log("Saving best model: epoch={}".format(self.epoch))
            self.best_loss = eval_metrics.avg['Loss']

        self.logger.scalar_summary(eval_metrics.avg, self.step, 'EVAL')

    def save(self):
        if self.args.model in dir(ace):
            for autoencoder in self.model.autoencoders:
                name = autoencoder.__class__.__name__.lower()
                filename = os.path.join(self.log_path, '{}-{}.pth'.format(name, self.epoch))
                torch.save({
                    'model': autoencoder.state_dict(),
                    'args': self.args
                }, filename)
            ckpt_num = self.args.ckpt_num*len(self.model.autoencoders)
        else:
            name = self.args.model
            filename = os.path.join(self.log_path, '{}-{}.pth'.format(name, self.epoch))
            torch.save({
                'model': self.model.state_dict(),
                'args': self.args
            }, filename)
            ckpt_num = self.args.ckpt_num

        pths = [(f, int(f[:-4].split("-")[-1])) for f in os.listdir(self.log_path) if f.endswith('.pth')]
        diff = len(pths) - ckpt_num
        if diff > 0:
            sorted_pths = sorted(pths, key=lambda tup: tup[1])
            for i in range(diff):
                os.remove(os.path.join(self.log_path, sorted_pths[i][0]))

    def adjust_learning_rate(self, milestone, factor=0.1):
        if self.epoch in milestone:
            self.lr *= factor
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.lr
