from pathlib import Path
from datetime import datetime
import time

import torch
import torch.nn.functional as F

from settings import PROJECT_ROOT
from trainer import Trainer
from common.logger import Logger
from common.torch_utils import get_model
from common.summary import EvaluationMetrics
import submodules.attacks as attacks


class Attacker(Trainer):
    """ Perform various adversarial attacks on a pretrained model

    Scheme generates Tensor, not Variable, input image is also tensor
    """
    def __init__(self, train_loader, val_loader, args, **kwargs):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.args = args
        self.model = get_model(args)
        self.step = 0
        self.cuda = self.args.cuda

        self.log_path = (
                PROJECT_ROOT / Path("experiments") /
                Path(datetime.now().strftime("%Y%m%d%H%M%S") + "-")
                ).as_posix()
        self.log_path = Path(self.get_dirname(self.log_path, args))
        if not Path.exists(self.log_path):
            Path(self.log_path).mkdir(parents=True, exist_ok=True)
        self.logger = Logger("attack", self.log_path, args.verbose)
        self.logger.log("Checkpoint files will be saved in {}".format(self.log_path))

        self.logger.add_level('ATTACK', 21)
        self.kwargs = kwargs

    def generate(self):
        self.model.eval()
        source = self.model
        scheme = getattr(attacks, self.args.attack)(source, self.args, **self.kwargs)

        eval_metrics = EvaluationMetrics(['Attack/Acc', 'AccDrop', 'Success', 'Conf', 'Time'])

        self.logger.log("Generate adversarial examples from validation set")
        for i, (images, labels) in enumerate(self.val_loader):
            self.step += 1
            if self.cuda:
                images = images.cuda()
                labels = labels.cuda()
            if self.args.half: images = images.half()

            st = time.time()
            outputs = self.model(self.to_var(images, self.cuda, True))
            outputs = outputs.float()
            _, preds = torch.max(outputs, 1)

            acc = (labels == preds.data).float().mean().item()

            adv_acc = 0
            acc_drop = 0
            success_rate = 0
            confidence = 0

            # Save intermediary images during generation
            info = {}
            info['logger'] = self.logger
            info['batch_no'] = i
            info['save_dir'] = "val"
            info['step'] = self.step

            adv_images, adv_labels = scheme.generate(images, labels, **info)

            outputs = self.model(self.to_var(adv_images, self.cuda, True))
            outputs = outputs.float()
            adv_prob, adv_preds = torch.max(F.softmax(outputs, dim=1), 1)
            adv_prob = torch.clamp(adv_prob, 0, 1)
            adv_acc = (labels == adv_preds.data).float().mean().item()
            acc_drop = (acc - adv_acc)

            if scheme.target is None:
                success = (adv_preds != preds)
            else:
                success = (adv_preds == self.to_var(adv_labels, self.cuda, True))
            success_rate = success.float().mean().item()
            confidence = 0 if not any(success.data) else adv_prob[success].float().mean().item()

            elapsed_time = time.time() - st
            batch_size = labels.size(0)
            eval_metrics.update('Attack/Acc', adv_acc, batch_size)
            eval_metrics.update('AccDrop', acc_drop, batch_size)
            eval_metrics.update('Success', success_rate, batch_size)
            eval_metrics.update('Conf', confidence, (confidence > 0)*batch_size)
            eval_metrics.update('Time', elapsed_time, batch_size)

            if self.step % self.args.log_step == 0 or self.step == len(self.val_loader):
                self.logger.scalar_summary(eval_metrics.avg, self.step, 'ATTACK')

        eval_metrics = EvaluationMetrics(['Attack/Acc', 'AccDrop', 'Success', 'Conf', 'Time'])

        self.logger.log("Generate adversarial examples from training set")
        self.step = 0
        for i, (images, labels) in enumerate(self.train_loader):
            self.step += 1
            if self.cuda:
                images = images.cuda()
                labels = labels.cuda()
            if self.args.half: images = images.half()

            st = time.time()
            outputs = self.model(self.to_var(images, self.cuda, True))
            outputs = outputs.float()
            _, preds = torch.max(outputs, 1)

            acc = (labels == preds.data).float().mean().item()

            adv_acc = 0
            acc_drop = 0
            success_rate = 0
            confidence = 0

            # Save intermediary images during generation
            info = {}
            info['logger'] = self.logger
            info['batch_no'] = i
            info['save_dir'] = "train"
            info['step'] = self.step

            adv_images, adv_labels = scheme.generate(images, labels, **info)
            outputs = self.model(self.to_var(adv_images, self.cuda, True))
            outputs = outputs.float()
            adv_prob, adv_preds = torch.max(F.softmax(outputs, dim=1), 1)
            adv_prob = torch.clamp(adv_prob, 0, 1)
            adv_acc = (labels == adv_preds.data).float().mean().item()
            acc_drop = (acc - adv_acc)

            if scheme.target is None:
                success = (adv_preds != preds)
            else:
                success = (adv_preds == self.to_var(adv_labels, self.cuda, True))
            success_rate = success.float().mean().item()
            confidence = 0 if not any(success.data) else adv_prob[success].float().mean().item()

            elapsed_time = time.time() - st
            batch_size = labels.size(0)
            eval_metrics.update('Attack/Acc', adv_acc, batch_size)
            eval_metrics.update('AccDrop', acc_drop, batch_size)
            eval_metrics.update('Success', success_rate, batch_size)
            eval_metrics.update('Conf', confidence, (confidence > 0)*batch_size)
            eval_metrics.update('Time', elapsed_time, batch_size)

            if self.step % self.args.log_step == 0 or self.step == len(self.train_loader):
                self.logger.scalar_summary(eval_metrics.avg, self.step, 'ATTACK')

    def to_var(self, x, cuda, volatile=False):
        """For CPU inference manual cuda setting is needed
        """
        if cuda:
            x = x.cuda()
        return torch.autograd.Variable(x, volatile=volatile)

    def get_dirname(self, path, args):
            path += "{}-".format(getattr(args, 'attack'))
            path += "{}-".format(getattr(args, 'model'))
            path += "{}-".format(getattr(args, 'seed'))
            return path
