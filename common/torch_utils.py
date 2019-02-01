import os
import re
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from settings import PROJECT_ROOT, MODEL_PATH_DICT
from nsml import NSML_NFS_OUTPUT


def get_model(args):
    import torchvision.models as torch_models
    import submodules.models as models
    import submodules.autoencoders as autoencoders
    import submodules.ace as ace

    model_name = args.model
    if model_name in dir(torch_models):
        model = getattr(models, model_name)(pretrained=True)
    else:
        model = getattr(models, model_name)(args)

    if model_name not in dir(torch_models) + dir(ace):
        if args.pretrained:
            # Default checkpoint name to model name
            ckpt_name = model_name if args.ckpt_name is None else args.ckpt_name

            if NSML_NFS_OUTPUT:
                path = os.path.join(NSML_NFS_OUTPUT, args.ckpt_dir)
                ckpt_name = MODEL_PATH_DICT.get(ckpt_name, ckpt_name)
            else:
                path = os.path.join(PROJECT_ROOT, args.ckpt_dir)

            found = False
            for root, _, filenames in os.walk(path):
                for filename in filenames:
                    if ckpt_name == filename:
                        found = True
                        path = os.path.join(root, filename)
                        break
                if found: break
            assert found, "Cannot find checkpoint file"

            ckpt = torch.load(path, map_location=lambda storage, loc: storage)
            model_state = ckpt['model']

            model_state_cpu = OrderedDict()
            for k in model_state.keys():
                if k.startswith("module."):
                    k_new = k[7:]
                    model_state_cpu[k_new] = model_state[k]
                else:
                    model_state_cpu[k] = model_state[k]
            model.load_state_dict(model_state_cpu)

        else:
            init_params(model, args=args)

    model.cuda() if args.cuda else model.cpu()
    if args.multigpu:
        model = nn.DataParallel(model)
    if args.half:
        model.half()
    model.eval()

    if model_name in dir(autoencoders):
        def compute_loss(model, images, labels):
            criterion = nn.MSELoss()
            if args.cuda:
                criterion = criterion.cuda()
            if args.half:
                criterion = criterion.half()
            outputs = model(images)
            loss = criterion(outputs, images)
            return None, loss
        return model, compute_loss

    elif model_name in dir(ace):
        def compute_loss(model, images, labels):
            criterion = nn.MSELoss()
            if args.cuda:
                criterion = criterion.cuda()
            if args.half:
                criterion = criterion.half()
            outputs = model(images, ae_only=True)
            recon_loss = criterion(outputs, images)

            outputs = model(images)
            class_loss = mixup(cross_entropy, model, images, labels, args)
            loss = args.recon_ratio*recon_loss + (1 - args.recon_ratio)*class_loss
            return outputs, loss
        return model, compute_loss

    else:
        def compute_loss(model, images, labels):
            outputs = model(images)
            loss = mixup(cross_entropy, model, images, labels, args)
            return outputs, loss
        return model, compute_loss


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


def init_params(model, args=None):
    """
    Initialize parameters in model

    Conv2d      : Xavier-Normal
    BatchNorm2d : Weight=1, Bias=0
    Linear      : Xavier-Normal

    """
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()

        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            m.bias.data.zero_()

        elif hasattr(m, '_modules'):
            for module in m._modules:
                try: init_params(module, args=args)
                except: continue


def mixup(criterion, model, images, labels, args=None):
    assert args is not None
    if args.mixup is None:
        outputs = model(images)
        loss = criterion(outputs, labels)
    else:
        lam = np.random.beta(args.mixup, args.mixup)
        idx = torch.randperm(args.batch_size)
        if args.cuda:
            idx = idx.cuda()
        
        images = lam*images + (1 - lam)*images[idx,:,:,:]
        outputs = model(images)
        loss = lam*criterion(outputs, labels) + (1 - lam)*criterion(outputs, labels[idx,:])

    return loss


def softmax(input, T=1):
    if T == 0:
        labels = torch.max(input, dim=1)[1]
        out = torch.cat([one_hot(label, input.size(1)).view(1,-1) for label in labels])
    else:
        out = F.softmax(input/T, dim=1)
    return out


def cross_entropy(output, target):
    out = torch.mean(torch.sum(-target*F.log_softmax(output, dim=1), dim=1))
    return out


def one_hot(label, num_classes=None):
    assert num_classes is not None
    label = torch.LongTensor([label])
    onehot = torch.FloatTensor(num_classes)
    onehot.zero_()
    onehot.scatter_(0, label, 1)
    return onehot
