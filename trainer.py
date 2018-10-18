import getpass
from pathlib import Path
from datetime import datetime
import time
import os
import inspect

import torch
import torch.nn as nn
import numpy as np
from scipy.stats import norm

from settings import PROJECT_ROOT
from common.logger import Logger
from common.torch_utils import to_np, get_optimizer, get_model, plot_grad_heatmap
from common.summary import EvaluationMetrics
from submodules.activation import CustomActivation
from submodules import attacks


class Trainer:
    """ Train and Validation with single GPU """
    def __init__(self, train_loader, val_loader, args):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.args = args
        self.model = get_model(args)
        self.epochs = args.epochs
        self.total_step = len(train_loader) * args.epochs
        self.step = 0
        self.epoch = 0
        self.start_epoch = 1
        self.lr = args.learning_rate
        self.best_acc = 0

        # Log
        self.log_path = (
                PROJECT_ROOT / Path("experiments") /
                Path(datetime.now().strftime("%Y%m%d%H%M%S") + "-")
                ).as_posix()
        self.log_path = Path(self.get_dirname(self.log_path, args))
        if not Path.exists(self.log_path):
            Path(self.log_path).mkdir(parents=True, exist_ok=True)
        self.logger = Logger("train", self.log_path, args.verbose)
        self.logger.log("Checkpoint files will be saved in {}".format(self.log_path))

        self.logger.add_level('STEP', 21, 'green')
        self.logger.add_level('EPOCH', 22, 'cyan')
        self.logger.add_level('EVAL', 23, 'yellow')

        self.criterion = nn.CrossEntropyLoss()
        if self.args.cuda:
            self.criterion = self.criterion.cuda()
        if args.half:
            self.model.half()
            self.criterion.half()

        params = self.model.parameters()
        self.optimizer = get_optimizer(args.optimizer, params, args)

    def train(self):
        self.eval()
        for self.epoch in range(self.start_epoch, self.args.epochs+1):
            self.adjust_learning_rate([int(self.args.epochs/2), int(self.args.epochs*3/4)], factor=0.1)
            self.train_epoch()
            self.eval()

            if self.epoch in [1, int(self.args.epochs/4), int(self.args.epochs/2), int(self.args.epochs*3/4)]:
                plot_grad_heatmap(self.model, self.args, self.logger, self.epoch)

        self.logger.writer.export_scalars_to_json(
            self.log_path.as_posix() + "/scalars-{}-{}-{}.json".format(
                self.args.model,
                self.args.seed,
                self.args.activation
            )
        )
        plot_grad_heatmap(self.model, self.args, self.logger, self.epoch)
        self.logger.writer.close()

    def train_epoch(self):
        self.model.train()
        eval_metrics = EvaluationMetrics(['Loss', 'Acc', 'Time'])

        for i, (images, labels) in enumerate(self.train_loader):
            st = time.time()
            self.step += 1
            images = torch.autograd.Variable(images)
            labels = torch.autograd.Variable(labels)
            if self.args.cuda:
                images = images.cuda()
                labels = labels.cuda()
            if self.args.half: images = images.half()

            outputs, loss = self.compute_loss(images, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            outputs = outputs.float()
            loss = loss.float()
            elapsed_time = time.time() - st

            _, preds = torch.max(outputs, 1)
            accuracy = (labels == preds.squeeze()).float().mean()

            batch_size = labels.size(0)
            eval_metrics.update('Loss', loss.data[0], batch_size)
            eval_metrics.update('Acc', accuracy.data[0], batch_size)
            eval_metrics.update('Time', elapsed_time, batch_size)

            if self.step % self.args.log_step == 0:
                self.logger.scalar_summary(eval_metrics.val, self.step, 'STEP')

        # Histogram of parameters
        for tag, p in self.model.named_parameters():
            tag = tag.split(".")
            tag = "Train_{}".format(tag[0]) + "/" + "/".join(tag[1:])
            try:
                self.logger.writer.add_histogram(tag, p.clone().cpu().data.numpy(), self.step)
                self.logger.writer.add_histogram(tag+'/grad', p.grad.clone().cpu().data.numpy(), self.step)
            except Exception as e:
                print("Check if variable {} is not used: {}".format(tag, e))

        self.logger.scalar_summary(eval_metrics.avg, self.step, 'EPOCH')


    def eval(self):
        self.model.eval()
        eval_metrics = EvaluationMetrics(['Loss', 'Acc', 'Time'])

        for i, (images, labels) in enumerate(self.val_loader):
            st = time.time()
            images = torch.autograd.Variable(images)
            labels = torch.autograd.Variable(labels)
            if self.args.cuda:
                images = images.cuda()
                labels = labels.cuda()
            if self.args.half: images = images.half()

            outputs, loss = self.compute_loss(images, labels)

            outputs = outputs.float()
            loss = loss.float()
            elapsed_time = time.time() - st

            _, preds = torch.max(outputs, 1)
            accuracy = (labels == preds.squeeze()).float().mean()

            batch_size = labels.size(0)
            eval_metrics.update('Loss', loss.data[0], batch_size)
            eval_metrics.update('Acc', accuracy.data[0], batch_size)
            eval_metrics.update('Time', elapsed_time, batch_size)

        # Save best model
        if eval_metrics.avg['Acc'] > self.best_acc:
            self.save()
            self.logger.log("Saving best model: epoch={}".format(self.epoch))
            self.best_acc = eval_metrics.avg['Acc']
            self.maybe_delete_old_pth(log_path=self.log_path.as_posix(), max_to_keep=1)

        self.logger.scalar_summary(eval_metrics.avg, self.step, 'EVAL')

    def get_dirname(self, path, args):
        path += "{}-".format(getattr(args, 'dataset'))
        path += "{}-".format(getattr(args, 'seed'))
        path += "{}-".format(getattr(args, 'model'))
        path += "{}-".format(getattr(args, 'activation'))
        path += "{}-".format(getattr(args, 'dropout'))
        return path

    def save(self, filename=None):
        if filename is None:
            filename = os.path.join(self.log_path, 'model-{}.pth'.format(self.epoch))
        torch.save({
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epoch': self.start_epoch,
            'best_acc': self.best_acc,
            'args': self.args
        }, filename)

    def load(self, filename=None):
        if filename is None: filename = self.log_path
        S = torch.load(filename) if self.args.cuda else torch.load(filename, map_location=lambda storage, location: storage)
        self.model.load_state_dict(S['model'])
        self.optimizer.load_state_dict(S['optimizer'])
        self.epoch = S['epoch']
        self.best_acc = S['best_acc']
        self.args = S['args']

    def infer(self):
        self.eval()

    def maybe_delete_old_pth(self, log_path, max_to_keep):
        """Model filename must end with xxx-xxx-[epoch].pth
        """
        # filename and time
        pths = [(f, int(f[:-4].split("-")[-1])) for f in os.listdir(log_path) if f.endswith('.pth')]
        if len(pths) > max_to_keep:
            sorted_pths = sorted(pths, key=lambda tup: tup[1])
            for i in range(len(pths) - max_to_keep):
                os.remove(os.path.join(log_path, sorted_pths[i][0]))

    def show_current_model(self):
        print("\n".join("{}: {}".format(k, v) for k, v in sorted(vars(self.args).items())))

        model_parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        total_params = np.sum([np.prod(p.size()) for p in model_parameters])

        print('%s\n\n'%(type(self.model)))
        print('%s\n\n'%(inspect.getsource(self.model.__init__)))
        print('%s\n\n'%(inspect.getsource(self.model.forward)))

        # Total 95
        print("*"*40 + "%10s" % self.args.model + "*"*45)
        print("*"*40 + "PARAM INFO" + "*"*45)
        print("-"*95)
        print("| %40s | %25s | %20s |" % ("Param Name", "Shape", "Number of Params"))
        print("-"*95)
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                print("| %40s | %25s | %20d |" % (name, list(param.size()), np.prod(param.size())))
        print("-"*95)
        print("Total Params: %d" % (total_params))
        print("*"*95)

    def adjust_learning_rate(self, milestone, factor=0.1):
        if self.epoch in milestone:
            self.lr *= factor
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.lr

    def compute_loss(self, images, labels):
        #TODO: check if RNN
        outputs = self.model(images)
        loss = self.criterion(outputs, labels)
        return outputs, loss


class AdvTrainer(Trainer):
    def __init__(self, train_loader, val_loader, args):
        super().__init__(train_loader, val_loader, args)
        self.kwargs = {}
        self.alpha = args.alpha
        self.attacker = getattr(attacks, self.args.attack)(self.model, self.args, **self.kwargs)

    def train_epoch(self):
        self.model.train()
        eval_metrics = EvaluationMetrics(['Loss', 'Acc', 'Time'])

        for i, (images, labels) in enumerate(self.train_loader):
            st = time.time()
            self.step += 1

            if self.args.cuda:
                images = images.cuda()
                labels = labels.cuda()
            adv_images, _ = self.attacker.generate(images, labels)
            if self.args.cuda: adv_images = adv_images.cuda()

            images = torch.autograd.Variable(images)
            adv_images = torch.autograd.Variable(adv_images)
            labels = torch.autograd.Variable(labels)
            if self.args.half:
                images = images.half()
                adv_images = adv_images.half()

            outputs_clean, loss_clean = self.compute_loss(images, labels)
            outputs_adv, loss_adv = self.compute_loss(adv_images, labels)
            images = torch.cat([images, adv_images], dim=0)
            labels = torch.cat([labels, labels])

            outputs = torch.cat([outputs_clean, outputs_adv], dim=0)
            loss = self.alpha * loss_clean + (1 - self.alpha) * loss_adv

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            outputs = outputs.float()
            loss = loss.float()
            elapsed_time = time.time() - st

            _, preds = torch.max(outputs, 1)
            accuracy = (labels == preds.squeeze()).float().mean()

            batch_size = labels.size(0)
            eval_metrics.update('Loss', loss.data[0], batch_size)
            eval_metrics.update('Acc', accuracy.data[0], batch_size)
            eval_metrics.update('Time', elapsed_time, batch_size)

            if self.step % self.args.log_step == 0:
                self.logger.scalar_summary(eval_metrics.val, self.step, 'STEP')

        # Histogram of parameters
        for tag, p in self.model.named_parameters():
            tag = tag.split(".")
            tag = "Train_{}".format(tag[0]) + "/" + "/".join(tag[1:])
            try:
                self.logger.writer.add_histogram(tag, p.clone().cpu().data.numpy(), self.step)
                self.logger.writer.add_histogram(tag+'/grad', p.grad.clone().cpu().data.numpy(), self.step)
            except Exception as e:
                print("Check if variable {} is not used: {}".format(tag, e))

        self.logger.scalar_summary(eval_metrics.avg, self.step, 'EPOCH')


class AETrainer(Trainer):
    def __init__(self, train_loader, val_loader, args):
        super().__init__(train_loader, val_loader, args)
        self.best_loss = np.inf
        self.criterion = nn.MSELoss()
        if self.args.cuda:
            self.criterion = self.criterion.cuda()
        if args.half:
            self.model.half()
            self.criterion.half()

    def train(self):
        self.eval()
        for self.epoch in range(self.start_epoch, self.args.epochs+1):
            self.adjust_learning_rate([int(self.args.epochs/2), int(self.args.epochs*3/4)], factor=0.1)
            self.train_epoch()
            self.eval()

        self.logger.writer.export_scalars_to_json(
            self.log_path.as_posix() + "/scalars-{}-{}-{}.json".format(
                self.args.model,
                self.args.seed,
                self.args.activation
            )
        )
        self.logger.writer.close()

    def train_epoch(self):
        self.model.train()
        eval_metrics = EvaluationMetrics(['Loss', 'Time'])

        for i, (images, _) in enumerate(self.train_loader):
            st = time.time()
            self.step += 1
            images = torch.autograd.Variable(images)
            if self.args.cuda:
                images = images.cuda()
            if self.args.half: images = images.half()

            outputs, loss = self.compute_loss(images)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            loss = loss.float()
            elapsed_time = time.time() - st

            batch_size = images.size(0)
            eval_metrics.update('Loss', loss.data[0], batch_size)
            eval_metrics.update('Time', elapsed_time, batch_size)

            if self.step % self.args.log_step == 0:
                self.logger.scalar_summary(eval_metrics.val, self.step, 'STEP')

        # Histogram of parameters
        for tag, p in self.model.named_parameters():
            tag = tag.split(".")
            tag = "Train_{}".format(tag[0]) + "/" + "/".join(tag[1:])
            try:
                self.logger.writer.add_histogram(tag, p.clone().cpu().data.numpy(), self.step)
                self.logger.writer.add_histogram(tag+'/grad', p.grad.clone().cpu().data.numpy(), self.step)
            except Exception as e:
                print("Check if variable {} is not used: {}".format(tag, e))

        self.logger.scalar_summary(eval_metrics.avg, self.step, 'EPOCH')

    def eval(self):
        self.model.eval()
        eval_metrics = EvaluationMetrics(['Loss', 'Time'])

        for i, (images, _) in enumerate(self.val_loader):
            st = time.time()
            images = torch.autograd.Variable(images)
            if self.args.cuda:
                images = images.cuda()
            if self.args.half: images = images.half()

            outputs, loss = self.compute_loss(images)

            loss = loss.float()
            elapsed_time = time.time() - st

            batch_size = images.size(0)
            eval_metrics.update('Loss', loss.data[0], batch_size)
            eval_metrics.update('Time', elapsed_time, batch_size)

        # Save best model
        if eval_metrics.avg['Loss'] < self.best_loss:
            self.save()
            self.logger.log("Saving best model: epoch={}".format(self.epoch))
            self.best_loss = eval_metrics.avg['Loss']
            self.maybe_delete_old_pth(log_path=self.log_path.as_posix(), max_to_keep=1)

        self.logger.scalar_summary(eval_metrics.avg, self.step, 'EVAL')

    def compute_loss(self, images):
        outputs = self.model(images)
        loss = self.criterion(outputs, images)
        return outputs, loss
