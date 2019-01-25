import time
import torch

from trainer import Trainer
import submodules.attacks as attacks

from common.logger import Logger
from common.summary import EvaluationMetrics
from common.torch_utils import get_model
from common.utils import get_dirname, show_current_model


class Attacker:
    def __init__(self, dataloader, args):
        self.loader = dataloader
        self.args = args

        self.model, self.compute_loss = get_model(args)
        if self.args.source is not None:
            args = self.args
            args.model = self.args.source
            args.ckpt_name = self.args.ckpt_src
            source, _ = get_model(args)
        else:
            source = self.model
        self.scheme = getattr(attacks, self.args.attack)(source, args)

        self.log_path = get_dirname(args)
        self.logger = Logger(self.args.mode, self.log_path, args.verbose)
        self.logger.log("Logs will be saved in {}".format(self.log_path))

        self.logger.add_level('ORIGINAL', 21, 'white')
        self.logger.add_level('ATTACKED', 22, 'yellow')

    def attack(self):
        show_current_model(self.model, self.args)

        eval_before = EvaluationMetrics(['Top1', 'Top5', 'Time'])
        eval_after = EvaluationMetrics(['Top1', 'Top5', 'Time'])

        for step, (images, labels) in enumerate(self.loader):
            st = time.time()
            if self.args.cuda:
                images = images.cuda()
                labels = labels.cuda()
            if self.args.half:
                images = images.half()

            outputs, _ = self.compute_loss(self.model, images, labels)
            elapsed_time = time.time() - st
            
            _, preds = torch.topk(outputs.float(), 5)
            top1 = (labels == preds[:,0]).float().mean().item()
            top5 = torch.sum((labels.unsqueeze(1).repeat(1,5) == preds).float(), 1).mean().item()

            batch_size = labels.size(0)
            eval_before.update('Top1', top1, batch_size)
            eval_before.update('Top5', top5, batch_size)
            eval_before.update('Time', elapsed_time, batch_size)

            st = time.time()
            adv_images = self.scheme.generate(images, labels)
            outputs, _ = self.compute_loss(self.model, adv_images, labels)
            elapsed_time = time.time() - st

            _, preds = torch.topk(outputs.float(), 5)
            top1 = (labels == preds[:,0]).float().mean().item()
            top5 = torch.sum((labels.unsqueeze(1).repeat(1,5) == preds).float(), 1).mean().item()

            batch_size = labels.size(0)
            eval_after.update('Top1', top1, batch_size)
            eval_after.update('Top5', top5, batch_size)
            eval_after.update('Time', elapsed_time, batch_size)

            if (step + 1) % self.args.log_step == 0:
                self.logger.scalar_summary(eval_before.avg, step + 1, 'ORIGINAL')
                self.logger.scalar_summary(eval_after.avg, step + 1, 'ATTACKED')
