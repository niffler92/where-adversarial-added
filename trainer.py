from argparse import Namespace
import time
import os
import torch
import numpy as np

import submodules.ace as ace
import submodules.attacks as attacks

from common.logger import Logger
from common.summary import EvaluationMetrics
from common.torch_utils import get_model, get_optimizer, softmax
from common.utils import get_dirname, show_current_model


class Trainer:
    def __init__(self, dataloader, args):
        self.loader = dataloader
        self.args = Namespace(**vars(args))

        self.model, self.compute_loss = get_model(self.args)

        # If adversarial training
        if self.args.adv_ratio is not None:
            if self.args.source is not None:
                args.model = self.args.source
                args.ckpt_name = self.args.ckpt_src
                self.source, _ = get_model(args)
            else:
                self.source = self.model
            self.scheme = getattr(attacks, self.args.attack)(self.source, args)

        self.epochs = self.args.epochs
        self.total_step = len(self.loader) * self.args.epochs
        self.step = 0
        self.epoch = 0
        self.start_epoch = 1
        self.lr = self.args.learning_rate

        self.log_path = get_dirname(self.args)
        self.logger = Logger(self.args.mode, self.log_path, self.args.verbose)
        self.logger.log("Logs will be saved in {}".format(self.log_path))

        self.logger.add_level('STEP', 21, 'white')
        self.logger.add_level('EPOCH', 22, 'cyan')
        self.logger.add_level('EVAL', 23, 'green')

        params = self.model.parameters()
        self.optimizer = get_optimizer(self.args.optimizer, params, self.args)

    def train(self):
        show_current_model(self.model, self.args)
        for self.epoch in range(self.start_epoch, self.args.epochs+1):
            self.adjust_learning_rate([int(self.args.epochs/2), int(self.args.epochs*3/4)], factor=0.1)
            self.train_epoch()
            self.save()
            if not self.args.no_eval:
                self.eval()

    def infer(self):
        show_current_model(self.model, self.args)
        self.eval()

    def train_epoch(self):
        eval_metrics = EvaluationMetrics(['Loss', 'Top1', 'Top5', 'Time'])

        for i, (images, labels) in enumerate(self.loader):
            st = time.time()
            self.step += 1
            if self.args.cuda:
                images = images.cuda()
                labels = labels.cuda()
            if self.args.half:
                images = images.half()

            self.model.train()
            outputs, loss = self.compute_loss(self.model, images, labels)

            # adversarial training with soft labels
            if self.args.adv_ratio is not None:
                self.source.eval()
                adv_images = self.scheme.generate(images, labels)
                adv_outputs = self.model(adv_images)

                if 'noise' in self.args.attack:
                    bs = self.args.batch_size
                    if 'linf' in self.args.attack:
                        cond_input, _ = torch.max(torch.abs(images).view(bs,-1), dim=1)
                        cond_output, _ = torch.max(torch.abs(outputs).view(bs,-1), dim=1)
                        cond_diff, _ = torch.max(torch.abs(adv_outputs - outputs).view(bs,-1), dim=1)

                    elif 'l2' in self.args.attack:
                        cond_input = torch.norm(images.view(bs,-1), dim=1)
                        cond_output = torch.norm(outputs.view(bs,-1), dim=1)
                        cond_diff = torch.norm((adv_outputs - outputs).view(bs,-1), dim=1)

                    cond_loss = (cond_diff/self.scheme.eps)*(cond_input/cond_output)
                    cond_loss = torch.mean(cond_loss)
                    cond_ratio = self.args.adv_ratio
                    loss = cond_ratio*cond_loss + (1 - cond_ratio)*loss

                else:
                    soft_labels = softmax(adv_outputs, self.args.distill_T)
                    if self.args.cuda:
                        soft_labels = soft_labels.cuda()
                    soft_ratio = self.args.distill_ratio
                    adv_labels = soft_ratio*soft_labels + (1 - soft_ratio)*labels

                    _, adv_loss = self.compute_loss(self.model, adv_images, adv_labels)
                    adv_ratio = self.args.adv_ratio
                    loss = adv_ratio*adv_loss + (1 - adv_ratio)*loss

            self.model.train()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            elapsed_time = time.time() - st
            loss = loss.item()

            if outputs is not None:
                _, targets = torch.max(labels, 1)
                _, preds = torch.topk(outputs, 5)
                top1 = (targets == preds[:,0]).float().mean().item()
                top5 = torch.sum((targets.unsqueeze(1).repeat(1,5) == preds).float(), 1).mean().item()
            else:
                top1 = np.nan
                top5 = np.nan

            eval_metrics.update('Loss', loss, self.args.batch_size)
            eval_metrics.update('Top1', top1, self.args.batch_size)
            eval_metrics.update('Top5', top5, self.args.batch_size)
            eval_metrics.update('Time', elapsed_time, self.args.batch_size)

            if self.step % self.args.log_step == 0:
                self.logger.scalar_summary(eval_metrics.val, self.step, 'STEP')

        self.logger.scalar_summary(eval_metrics.avg, self.step, 'EPOCH')

    def eval(self):
        self.model.eval()
        eval_metrics = EvaluationMetrics(['Loss', 'Top1', 'Top5', 'Time'])

        for i, (images, labels) in enumerate(self.loader):
            st = time.time()
            self.step += 1
            if self.args.cuda:
                images = images.cuda()
                labels = labels.cuda()
            if self.args.half:
                images = images.half()

            outputs, loss = self.compute_loss(self.model, images, labels)

            elapsed_time = time.time() - st
            loss = loss.item()

            if outputs is not None:
                _, targets = torch.max(labels, 1)
                _, preds = torch.topk(outputs, 5)
                top1 = (targets == preds[:,0]).float().mean().item()
                top5 = torch.sum((targets.unsqueeze(1).repeat(1,5) == preds).float(), 1).mean().item()
            else:
                top1 = np.nan
                top5 = np.nan

            eval_metrics.update('Loss', loss, self.args.batch_size)
            eval_metrics.update('Top1', top1, self.args.batch_size)
            eval_metrics.update('Top5', top5, self.args.batch_size)
            eval_metrics.update('Time', elapsed_time, self.args.batch_size)

            # No step summaries when training
            if self.step % self.args.log_step == 0 and self.args.mode == 'infer':
                self.logger.scalar_summary(eval_metrics.avg, self.step, 'EVAL')

        self.logger.scalar_summary(eval_metrics.avg, self.step, 'EVAL')

    def save(self):
        if self.args.model in dir(ace):
            if self.args.multigpu:
                autoencoders = self.model.module.autoencoders
                classifiers = self.model.module.classifiers
            else:
                autoencoders = self.model.autoencoders
                classifiers = self.model.classifiers

            for autoencoder in autoencoders:
                name = autoencoder.__class__.__name__.lower()
                filename = os.path.join(self.log_path, '{}-{}.pth'.format(name, self.epoch))
                torch.save({
                    'model': autoencoder.state_dict(),
                    'args': self.args
                }, filename)
            
            if self.args.fine_tune:
                for classifier in classifiers:
                    name = classifier.__class__.__name__.lower()
                    filename = os.path.join(self.log_path, '{}-{}.pth'.format(name, self.epoch))
                    torch.save({
                        'model': classifier.state_dict(),
                        'args': self.args
                    }, filename)

        else:
            name = self.args.model
            filename = os.path.join(self.log_path, '{}-{}.pth'.format(name, self.epoch))
            torch.save({
                'model': self.model.state_dict(),
                'args': self.args
            }, filename)

    def adjust_learning_rate(self, milestone, factor=0.1):
        if self.epoch in milestone:
            self.lr *= factor
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.lr
