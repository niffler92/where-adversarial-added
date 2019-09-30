import os
from argparse import Namespace
from collections import OrderedDict
import re

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from settings import PROJECT_ROOT, LOAD_DIR
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


def get_checkpoint(args=None):
    if args.ckpt_name:
        root = os.path.join(LOAD_DIR, args.ckpt_name)
    else:
        root = os.path.join(LOAD_DIR, args.dataset, args.model)
    assert os.path.isfile(root), "Checkpoint file does not exist."
    return root


def get_model(args):
    import submodules.models as models
    import submodules.autoencoders as autoencoders

    args_orig = Namespace(**vars(args))
    pretrained = args.pretrained

    autoencoders = sorted(name for name in dir(autoencoders))
    for ae in autoencoders:
        if ae + '_' in args.model and not args.autoencoder:
            model = getattr(models, args.model)(args)
            model.cuda() if args.cuda else model.cpu()
            if args.multigpu:
                model = nn.DataParallel(model, device_ids=list(range(args.multigpu)))
            if args.half:
                model.half()
            return model

    if pretrained and (args.dataset != 'ImageNet' or args.autoencoder):
        path = get_checkpoint(args)
        ckpt = torch.load(path, map_location=lambda storage, loc: storage)
        model_state = ckpt['model']
        args = ckpt['args']

    if pretrained and args.dataset == 'ImageNet' and not args.autoencoder:
        import torchvision.models as models
        model = getattr(models, args.model)(pretrained=True)

    else:
        import submodules.models as models
        model = getattr(models, args.model)(args)

    if pretrained and (args.dataset != 'ImageNet' or args.autoencoder):
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
            getattr(nn.init, args.conv_weight_init)(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()

        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

        elif isinstance(m, nn.Linear):
            getattr(nn.init, args.conv_weight_init)(m.weight)
            m.bias.data.zero_()

        elif hasattr(m, '_modules'):
            for module in m._modules:
                try: init_params(module, args=args)
                except: continue
