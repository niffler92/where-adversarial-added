import os
from argparse import Namespace
from collections import OrderedDict
import re

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import nsml

from common.utils import get_checkpoint
from settings import PROJECT_ROOT
from dataloader import get_loader


def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    if not isinstance(x, Variable):
        x = Variable(x, volatile=volatile)
    return x


def to_np(x):
    if isinstance(x, Variable):
        x = x.data
    return x.cpu().numpy()


def adjust_learning_rate(lr, optimizer, epoch):
    lr = lr * (0.5 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def get_optimizer(optimizer, params, args):
    optimizer = optimizer.lower()
    assert optimizer in ['sgd', 'adam', 'rmsprop', 'sgd_nn', 'adadelta']
    params = filter(lambda p: p.requires_grad, params)

    if optimizer == 'sgd':
        return torch.optim.SGD(
            params, args.learning_rate, momentum=args.momentum,
            nesterov=True, weight_decay=args.weight_decay
        )
    elif optimizer == 'adam':
        return torch.optim.Adam(
            params, args.learning_rate,
            weight_decay=args.weight_decay
        )
    elif optimizer == 'rmsprop':
        return torch.optim.RMSprop(
            params, args.learning_rate,
            weight_decay=args.weight_decay
        )
    elif optimizer == 'sgd_nn':
        return torch.optim.SGD(
            params, args.learning_rate, momentum=0,
            nesterov=False, weight_decay=0
        )
    elif optimizer == 'adadelta':
        return torch.optim.Adadelta(
            params, args.learning_rate,
            weight_decay=args.weight_decay
        )


def get_model(args):
    args_orig = Namespace(**vars(args))  # copy
    pretrained = args.pretrained
    autoencoder = False

    #FIXME: give attr for autoencoder networks
    if 'segnet' in args.model or 'unet' in args.model:
        autoencoder = True

    if '_' in args.model and autoencoder:
        import submodules.models as models
        model = getattr(models, args.model)(args)
        model.cuda() if args.cuda else model.cpu()
        if args.multigpu:
            model = nn.DataParallel(model, device_ids=list(range(args.multigpu)))
        if args.half:
            model.half()

        return model

    if pretrained and (args.dataset != 'ImageNet' or autoencoder):
        path = get_checkpoint(args)
        ckpt = torch.load(path, map_location=lambda storage, loc: storage)
        model_state = ckpt['model']
        args = ckpt['args']

    if pretrained and args.dataset == 'ImageNet' and not autoencoder:
        import torchvision.models as models
        model = getattr(models, args.model)(pretrained=True)

    else:
        import submodules.models as models
        model = getattr(models, args.model)(args)

    if pretrained and (args.dataset != 'ImageNet' or autoencoder):
        model_state_cpu = OrderedDict()
        for k in model_state.keys():
            if k.startswith("module."):
                k_new = k[7:]
                model_state_cpu[k_new] = model_state[k]
            else:
                model_state_cpu[k] = model_state[k]
        model.load_state_dict(model_state_cpu)
    if not pretrained:
        init_params(model, args=args)

    args = args_orig
    model.cuda() if args.cuda else model.cpu()
    if args.multigpu:
        model = nn.DataParallel(model, device_ids=list(range(args.multigpu)))
    if args.half:
        model.half()

    return model


def init_params(model, args=None):
    """
    Initialize parameters in model

    Conv2d      : Xavier-Normal
    BatchNorm2d : Weight=1, Bias=0
    Linear      : Xavier-Normal

    """
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            #nn.init.xavier_normal(m.weight)
            getattr(nn.init, args.conv_weight_init)(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()

        elif isinstance(m, nn.BatchNorm2d):
            if not args.no_bn and not args.no_affine:
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        elif isinstance(m, nn.Linear):
            #nn.init.xavier_normal(m.weight)
            getattr(nn.init, args.conv_weight_init)(m.weight)
            m.bias.data.zero_()

        elif hasattr(m, '_modules'):
            for module in m._modules:
                try: init_params(module, args=args)
                except: continue


def plot_grad_heatmap(model, args, logger, epoch):
    """requires NSML and uses visdom
    model must have gradient hook in order to get gradients
    """
    if not nsml.IS_ON_NSML:
        logger.log("Not on NSML. No heatmap will be generated", 'WARNING')
        return

    if not hasattr(model, "save_grad"):
        logger.log("No gradient hook on model. No heatmap will be generated", 'WARNING')
        return

    def move_axis(sample):
        if len(sample.shape) == 4:
            return np.moveaxis(sample, [1, 2, 3], [3, 1, 2])
        elif len(sample.shape) == 3:
            return np.moveaxis(sample, [0, 1, 2], [2, 0, 1])

    def get_gradient(sample, label, model):
        sample = sample.cuda()
        label = label.cuda()
        model.cuda()

        model.set_grad()
        out = model(sample)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(out, label)

        if 'input' in model.grads and model.grads['input'].grad:
            for key in model.grads.keys():
                model.grads[key].grad.data.zero_()
        loss.backward()

        return out, model.grads

    model.train()
    abs_grads = {}

    _, val_loader = get_loader(args.dataset, batch_size=64, num_workers=args.workers)
    for x_train, y_train in val_loader:
        x_train, y_train = Variable(x_train, requires_grad=True), Variable(y_train)
        if args.cuda:
            x_train, y_train = x_train.cuda(), y_train.cuda()

        _, grads = get_gradient(x_train, y_train, model)

        for key in grads.keys():
            abs_grad = np.abs(move_axis(to_np(grads[key]))).sum((0, 3))

            if key in abs_grads:
                abs_grads[key] += abs_grad
            else:
                abs_grads[key] = abs_grad

    for key in grads.keys():
        abs_grads[key] = torch.from_numpy(abs_grads[key])
    logger.heatmap_summary(abs_grads, step="Gradient_val-{}".format(epoch))
