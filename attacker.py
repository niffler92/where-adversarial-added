import getpass
from pathlib import Path
from datetime import datetime
import time
import os
from collections import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from settings import PROJECT_ROOT
from trainer import Trainer
from common.logger import Logger
from common.torch_utils import to_np, to_var, get_optimizer, get_model
from common.attack_utils import get_artifact
from common.summary import EvaluationMetrics
import submodules.attacks as attacks
import submodules.defenses as defenses


class Attacker(Trainer):
    """ Perform various adversarial attacks on a pretrained model

    Scheme generates Tensor, not Variable, input image is also tensor
    """
    def __init__(self, val_loader, args, **kwargs):
        self.val_loader = val_loader
        self.args = args
        self.model = get_model(args)
        self.step = 0
        self.cuda = self.args.cuda

        self.log_path = (
                PROJECT_ROOT / Path(args.save_dir) /
                Path(datetime.now().strftime("%Y%m%d%H%M%S") + "-")
                ).as_posix()
        self.log_path = Path(self.get_dirname(self.log_path, args))
        if not Path.exists(self.log_path):
            Path(self.log_path).mkdir(parents=True, exist_ok=True)
        self.logger = Logger("attack", self.log_path, args.verbose)
        self.logger.log("Checkpoint files will be saved in {}".format(self.log_path))

        self.logger.add_level('ATTACK', 21)
        self.logger.add_level('ARTIFACT', -1)

        self.kwargs = kwargs
        if args.log_artifact or args.domain_restrict:
            self.artifact = get_artifact(self.model, val_loader, args)
            self.kwargs['artifact'] = self.artifact

    def attack(self):
        self.model.eval()
        source = self.model
        if self.args.source is not None:
            target = self.args.ckpt_name
            self.args.model = self.args.source
            self.args.ckpt_name = self.args.ckpt_src
            source = get_model(self.args)
            self.logger.log("Transfer attack from {} -> {}".format(self.args.ckpt_src, target))
        schemes = getattr(attacks, self.args.attack)(source, self.args, **self.kwargs)
        if not isinstance(schemes, Iterable): schemes = [schemes]

        eval_metrics = EvaluationMetrics(['Attack/Acc', 'AccDrop', 'Success', 'Conf', 'Time'])
        if self.args.log_artifact:
            artifact_metrics = EvaluationMetrics(['Artifact/All', 'Artifact/Success'])
            artifact_heatmap = EvaluationMetrics(['Artifact/Diff', 'Artifact/Count'])

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

            acc = (labels == preds.data).float().mean()

            adv_acc = 0
            acc_drop = 0
            success_rate = 0
            confidence = 0

            for scheme in schemes:
                adv_images, adv_labels = scheme.generate(images, labels)
                outputs = self.model(self.to_var(adv_images, self.cuda, True))
                outputs = outputs.float()
                adv_prob, adv_preds = torch.max(F.softmax(outputs, dim=1), 1)
                adv_prob = torch.clamp(adv_prob, 0, 1)
                _adv_acc = (labels == adv_preds.data).float().mean()
                _acc_drop = acc - _adv_acc

                if scheme.target is None:
                    success = (adv_preds != preds)
                else:
                    success = (adv_preds == self.to_var(adv_labels, self.cuda, True))
                _success_rate = success.float().mean().data[0]
                _confidence = 0 if not any(success.data) else adv_prob[success].float().mean().data[0]

                if self.args.aggregate == 'success':
                    if success_rate <= _success_rate:
                        success_rate = _success_rate
                        if confidence <= _confidence:
                            confidence = _confidence
                            acc_drop = _acc_drop
                            adv_acc = _adv_acc
                elif self.args.aggregate == 'accdrop':
                    if acc_drop <= _acc_drop:
                        adv_acc = _adv_acc
                        acc_drop = _acc_drop
                        confidence = _confidence
                        succes_rate = _success_rate
                else:
                    raise NotImplementedError

                if self.args.log_artifact:
                    diff = torch.sum(torch.abs(images - adv_images), dim=1)
                    artifact_heatmap.update('Artifact/Diff', diff.float(), labels.size(0))

                    diff = ~torch.lt(diff, 1e-5)
                    artifact_heatmap.update('Artifact/Count', diff.float(), labels.size(0))

                    n_pix = torch.sum(diff)
                    if n_pix != 0:
                        artifact = self.artifact.unsqueeze(0).repeat(diff.size(0),1,1)
                        artifact = artifact[diff].sum()
                        artifact_metrics.update('Artifact/All', artifact/n_pix, n_pix)

                    diff = diff * success.data.view(-1,1,1)
                    n_pix = torch.sum(diff)
                    if n_pix != 0:
                        artifact = self.artifact.unsqueeze(0).repeat(diff.size(0),1,1)
                        artifact = artifact[diff].sum()
                        artifact_metrics.update('Artifact/Success', artifact/n_pix, n_pix)

            elapsed_time = time.time() - st
            batch_size = labels.size(0)
            eval_metrics.update('Attack/Acc', float(adv_acc), batch_size)
            eval_metrics.update('AccDrop', float(acc_drop), batch_size)
            eval_metrics.update('Success', float(success_rate), batch_size)
            eval_metrics.update('Conf', float(confidence), float(confidence > 0)*batch_size)
            eval_metrics.update('Time', elapsed_time, batch_size)

            if self.step % self.args.avg_step == 0 or self.step == len(self.val_loader):
                self.logger.scalar_summary(eval_metrics.avg, self.step, 'ATTACK')
                if self.args.log_artifact:
                    self.logger.scalar_summary(artifact_metrics.avg, self.step, 'ARTIFACT')
            if self.args.log_artifact and self.step % self.args.img_log_step == 0:
                heatmaps = {
                    'Diff': torch.mean(artifact_heatmap.avg['Artifact/Diff'], dim=0),
                    'Count': torch.sum(artifact_heatmap.sum['Artifact/Count'], dim=0)
                }
                self.logger.heatmap_summary(heatmaps, self.step)

    def to_var(self, x, cuda, volatile=False):
        """For CPU inference manual cuda setting is needed
        """
        if cuda:
            x = x.cuda()
        return torch.autograd.Variable(x, volatile=volatile)
