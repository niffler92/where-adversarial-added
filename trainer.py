import time
import os

import torch
import torch.nn as nn
import numpy as np

from common.logger import Logger
from common.summary import EvaluationMetrics
from common.torch_utils import get_model, get_optimizer
from common.utils import get_dirname, show_current_model
import nsml


class Trainer:
    def __init__(self, train_loader, val_loader, args):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.args = args
        self.model, self.compute_loss = get_model(args)
        self.epochs = args.epochs

        self.total_step = len(train_loader) * args.epochs
        self.step = 0
        self.epoch = 0
        self.start_epoch = 1
        self.lr = args.learning_rate
        self.best_loss = np.inf

        self.log_path = get_dirname(args)
        self.logger = Logger("train", self.log_path, args.verbose)
        self.logger.log("Checkpoint files will be saved in {}".format(self.log_path))

        self.logger.add_level('STEP', 21, 'green')
        self.logger.add_level('EPOCH', 22, 'cyan')
        self.logger.add_level('EVAL', 23, 'yellow')

        params = self.model.parameters()
        self.optimizer = get_optimizer(args.optimizer, params, args)

    def train(self):
        show_current_model(self.model, self.args)
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

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            elapsed_time = time.time() - st
            loss = loss.item()
            if outputs is not None:
                outputs = outputs.float()
                _, preds = torch.max(outputs, 1)
                accuracy = (labels == preds.squeeze()).float().mean().item()
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

            elapsed_time = time.time() - st
            loss = loss.item()
            if outputs is not None:
                outputs = outputs.float()
                _, preds = torch.max(outputs, 1)
                accuracy = (labels == preds.squeeze()).float().mean().item()
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
        filename = os.path.join(self.log_path, 'model-{}.pth'.format(self.epoch))
        torch.save({
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epoch': self.epoch,
            'best_loss': self.best_loss,
            'args': self.args
        }, filename)

        pths = [(f, int(f[:-4].split("-")[-1])) for f in os.listdir(self.log_path) if f.endswith('.pth')]
        diff = len(pths) - self.args.ckpt_num
        if diff > 0:
            sorted_pths = sorted(pths, key=lambda tup: tup[1])
            for i in range(diff):
                os.remove(os.path.join(self.log_path, sorted_pths[i][0]))

    def adjust_learning_rate(self, milestone, factor=0.1):
        if self.epoch in milestone:
            self.lr *= factor
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.lr
