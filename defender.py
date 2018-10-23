import getpass
from pathlib import Path
from datetime import datetime
import time
import os
from collections import Iterable
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from settings import PROJECT_ROOT
from attacker import Attacker
from common.logger import Logger
from common.torch_utils import to_np, to_var, get_optimizer, get_model
from common.attack_utils import get_artifact
from common.summary import EvaluationMetrics
import submodules.attacks as attacks
import submodules.defenses as defenses
from dataloader import denormalize, normalize


class Defender(Attacker):
    """ Perform various adversarial attacks and defense on a pretrained model
    Scheme generates Tensor, not Variable
    """
    def __init__(self, val_loader, args, **kwargs):
        super().__init__(val_loader, args, **kwargs)
        self.logger.name = "defense"
        self.logger.add_level("DEFENSE", 22, 'yellow')
        self.logger.add_level("TEST", 23, 'white')
        self.logger.add_level("DIST", 11, 'cyan')

    def defend(self):
        self.model.eval()
        defense_scheme = getattr(defenses, self.args.defense)(self.model, self.args, **self.kwargs)
        source = self.model
        if self.args.source is not None and (self.args.ckpt_name != self.args.ckpt_src):
            target = self.args.ckpt_name
            self.args.model = self.args.source
            self.args.ckpt_name = self.args.ckpt_src
            source = get_model(self.args)
            self.logger.log("Transfer attack from {} -> {}".format(self.args.ckpt_src, target))
        attack_scheme = getattr(attacks, self.args.attack)(source, self.args, **self.kwargs)

        eval_metrics = EvaluationMetrics(['Test/Acc', 'Test/Top5', 'Test/Time', 'Def-Test/Acc', 'Def-Test/Time'])
        attack_metrics = EvaluationMetrics(['Attack/Acc', 'Attack/Top5', 'Attack/Time'])
        defense_metrics = EvaluationMetrics(['Defense/Acc', 'Defense/Top5', 'Defense/Time'])
        dist_metrics = EvaluationMetrics(['L0', 'L1', 'L2', 'Li'])

        if self.args.log_artifact:
            self.artifact_metrics = EvaluationMetrics(['Artifact/All', 'Artifact/Success'])
            self.artifact_heatmap = EvaluationMetrics(['Artifact/Diff', 'Artifact/Count'])

        for i, (images, labels) in enumerate(self.val_loader):
            self.step += 1
            if self.cuda:
                images = images.cuda()
                labels = labels.cuda()
            if self.args.half: images = images.half()

            # Inference
            st = time.time()
            outputs = self.model(self.to_var(images, self.cuda, True))
            outputs = outputs.float()
            _, preds = torch.topk(outputs, 5)

            acc = (labels == preds.data[:,0]).float().mean()
            top5 = torch.sum((labels.unsqueeze(1).repeat(1,5) == preds.data).float(), dim=1).mean()
            eval_metrics.update('Test/Acc', acc, labels.size(0))
            eval_metrics.update('Test/Top5', top5, labels.size(0))
            eval_metrics.update('Test/Time', time.time()-st, labels.size(0))

            # Attacker
            st = time.time()
            adv_images, adv_labels = attack_scheme.generate(images, labels)
            if isinstance(adv_images, Variable):  # FIXME - Variable in Variable out for all methods
                adv_images = adv_images.data
            attack_metrics.update('Attack/Time', time.time()-st, labels.size(0))

            # Lp distance
            diff = torch.abs(denormalize(adv_images, self.args.dataset) - denormalize(images, self.args.dataset))
            L0 = torch.sum((torch.sum(diff, dim=1) > 1e-3).float().view(labels.size(0), -1), dim=1).mean()
            diff = diff.view(labels.size(0), -1)
            L1 = torch.norm(diff, p=1, dim=1).mean()
            L2 = torch.norm(diff, p=2, dim=1).mean()
            Li = torch.max(diff, dim=1)[0].mean()
            dist_metrics.update('L0', L0, labels.size(0))
            dist_metrics.update('L1', L1, labels.size(0))
            dist_metrics.update('L2', L2, labels.size(0))
            dist_metrics.update('Li', Li, labels.size(0))

            # Defender
            st = time.time()
            def_images, def_labels = defense_scheme.generate(adv_images, adv_labels)
            if isinstance(def_images, Variable):  # FIXME - Variable in Variable out for all methods
                def_images = def_images.data
            defense_metrics.update('Defense/Time', time.time()-st, labels.size(0))
            self.calc_stats('Attack', adv_images, images, adv_labels, labels, attack_metrics)
            self.calc_stats('Defense', def_images, images, def_labels, labels, defense_metrics)

            # Defense-Inference for shift of original image
            st = time.time()
            def_images_org, _ = defense_scheme.generate(images, labels)
            if isinstance(def_images_org, Variable):  # FIXME - Variable in Variable out for all methods
                def_images_org = def_images_org.data
            outputs = self.model(self.to_var(def_images_org, self.cuda, True))
            outputs = outputs.float()
            _, preds = torch.max(outputs, 1)
            acc = (labels == preds.data).float().mean()
            eval_metrics.update('Def-Test/Acc', acc, labels.size(0))
            eval_metrics.update('Def-Test/Time', time.time()-st, labels.size(0))

            if self.step % self.args.avg_step == 0 or self.step == len(self.val_loader):
                self.logger.scalar_summary(eval_metrics.avg, self.step, 'TEST')
                self.logger.scalar_summary(attack_metrics.avg, self.step, 'ATTACK')
                self.logger.scalar_summary(defense_metrics.avg, self.step, 'DEFENSE')
                self.logger.scalar_summary(dist_metrics.avg, self.step, 'DIST')
                if self.args.log_artifact:
                    self.logger.scalar_summary(self.artifact_metrics.avg, self.step, 'ARTIFACT')
            if self.args.log_artifact and self.step % self.args.img_log_step == 0:
                heatmaps = {
                    'Diff': torch.mean(self.artifact_heatmap.avg['Artifact/Diff'], dim=0),
                    'Count': torch.sum(self.artifact_heatmap.sum['Artifact/Count'], dim=0)
                }
                self.logger.heatmap_summary(heatmaps, self.step)

            if self.step % self.args.img_log_step == 0:
                image_dict = {
                    'Original': to_np(denormalize(images, self.args.dataset))[0],
                    'Attacked': to_np(denormalize(adv_images, self.args.dataset))[0],
                    'Defensed': to_np(denormalize(def_images, self.args.dataset))[0],
                    'Perturbation': to_np(denormalize(images - adv_images, self.args.dataset))[0]
                }
                self.logger.image_summary(image_dict, self.step)

            defense_rate = eval_metrics.avg['Test/Acc'] - defense_metrics.avg['Defense/Acc']
            if eval_metrics.avg['Test/Acc'] - attack_metrics.avg['Attack/Acc']:
                defense_rate /= eval_metrics.avg['Test/Acc'] - attack_metrics.avg['Attack/Acc']
            else:
                defense_rate = 0
            defense_rate = 1 - defense_rate

            defense_top5 = eval_metrics.avg['Test/Top5'] - defense_metrics.avg['Defense/Top5']
            if eval_metrics.avg['Test/Top5'] - attack_metrics.avg['Attack/Top5']:
                defense_top5 /= eval_metrics.avg['Test/Top5'] - attack_metrics.avg['Attack/Top5']
            else:
                defense_top5 = 0
            defense_top5 = 1 - defense_top5

            self.logger.log("Defense Rate Top1: {:5.3f} | Defense Rate Top5: {:5.3f}".format(defense_rate, defense_top5), 'DEFENSE')

    def calc_stats(self, method, gen_images, images, gen_labels, labels, metrics):
        """gen_images: Generated from attacker or defender
        Currently just calculating acc and artifact
        """
        # Wrong semantic meaning on adv_label............ Hmmm
        # adv_labels: Target label if target else original label
        # FIXME Need Success and Confidence to be calculated. What is the criterion of successful defense?
        success_rate = 0

        if not isinstance(gen_images, Variable):
            gen_images = self.to_var(gen_images.clone(), self.cuda, True)
        gen_outputs = self.model(gen_images)
        gen_outputs = gen_outputs.float()
        _, gen_preds = torch.topk(F.softmax(gen_outputs, dim=1), 5)

        if isinstance(gen_preds, Variable):
            gen_preds = gen_preds.data
        gen_acc = (labels == gen_preds[:,0]).float().mean()
        gen_top5 = torch.sum((labels.unsqueeze(1).repeat(1,5) == gen_preds).float(), dim=1).mean()
        #print("{} - True label: {}, Generated label: {}".format(method, labels[0], gen_preds[0]))

        if method == "Attack" and self.args.log_artifact:
            diff = torch.sum(torch.abs(images - gen_images), dim=1)
            self.artifact_heatmap.update('Artifact/Diff', diff.float(), labels.size(0))

            diff = ~torch.lt(diff, 1e-5)
            self.artifact_heatmap.update('Artifact/Count', diff.float(), labels.size(0))

            n_pix = torch.sum(diff)
            if n_pix != 0:
                artifact = self.artifact.unsqueeze(0).repeat(diff.size(0),1,1)
                artifact = artifact[diff].sum()
                self.artifact_metrics.update('Artifact/All', artifact/n_pix, n_pix)

        metrics.update('{}/Acc'.format(method), gen_acc, labels.size(0))
        metrics.update('{}/Top5'.format(method), gen_top5, labels.size(0))

    def to_var(self, x, cuda, volatile=False):
        """For CPU inference manual cuda setting is needed
        """
        if cuda:
            x = x.cuda()
        return torch.autograd.Variable(x, volatile=volatile)
