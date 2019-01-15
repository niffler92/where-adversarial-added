import numpy as np
import torch
import torch.nn as nn
import copy


def get_cb_grid(width, height, kernel_size, stride, padding=0):
    """Returns checkerboard overlap grid
    Only for width = height, kernel_size[0] = kernel_size[1], stride[0] = stride[1]
    """
    width += 2 * padding
    height += 2 * padding
    grid = np.zeros([width, height])
    kernel = np.ones([kernel_size, kernel_size])

    for r in range(0, width-kernel_size+1, stride):
        for c in range(0, height-kernel_size+1, stride):
            grid[r:r+kernel_size, c:c+kernel_size] += 1
    return grid[padding:-padding, padding:-padding] if padding else grid


def get_gradient(sample, label, model):
    """For models with name_g, retrieves all registered gradients.
    Others only get input gradients
    """
    sample = sample.cuda()
    label = label.cuda()
    model.cuda()

    for p in model.parameters():
        if p.requires_grad:
            p.requires_grad = False

    if hasattr(model, 'grad'):
        model.set_grad()
    else:
        x_grad  = None
        def hook_nonleaf_grad(grad):
            nonlocal x_grad
            x_grad = grad
        sample.register_hook(hook_nonleaf_grad)

    out = model(sample)
    criterion = nn.CrossEntropyLoss()
    loss = criterion(out, label)

    if hasattr(model, 'grads') and 'input' in model.grads and model.grads['input'].grad:
        for key in model.grads.keys():
            model.grads[key].grad.data.zero_()
    elif sample.grad:
        sample.grad.data.zero_()
    else:
        raise ValueError("No registered hook on model")

    loss.backward()

    return out, {'input': sample.grad} if sample.grad else model.grads


def init_rf(model):
    # FIXME: Temporary fix only for ResNet
    scale = 0.1
    is_bottleneck = "bottleneck" in str(model.modules().__next__()).lower()
    for m in model.modules():
        if hasattr(m, "weight"):
            m.weight.data = torch.ones_like(m.weight).data*scale
            # check if shortcut
            ks = getattr(m, "kernel_size", lambda: None)
            s = getattr(m, "stride", lambda: None)
            if ks == (1,1) and s == (2,2):
                m.weight.data *= scale
                if is_bottleneck:
                    m.weight.data *= scale
        if hasattr(m, "bias"):
            if m.bias is not None:
                m.bias.data.zero_()
        if hasattr(m, "running_mean"):
            m.training = False
            if m.running_mean is not None:
                m.running_mean.zero_()
        if hasattr(m, "running_var"):
            m.training = False
            if m.running_var is not None:
                m.running_var.fill_(1)
        if hasattr(m, '_modules'):
            for module in m._modules:
                try: init_rf(module)
                except: continue

def get_rf(model, images, args):
    model_rf = copy.deepcopy(model)
    input = torch.ones_like(images)
    input = torch.autograd.Variable(input, requires_grad=True)
    if args.cuda:
        input = input.cuda()

    init_rf(model_rf)
    model_rf.zero_grad()
    model_rf.set_grad()
    out = model_rf(input)
    out.backward(torch.ones_like(out))

    grad = torch.mean(torch.abs(model_rf.grads['input']), dim=1)
    #grad = torch.mean(grad, dim=0).data
    #grad /= torch.max(grad)

    return grad


def get_artifact(model, dataloader, args):
    images, _ = iter(dataloader).next()
    artifact = get_rf(model, images, args)
    artifact = (artifact - torch.min(artifact))/(torch.max(artifact) - torch.min(artifact))

    k = np.prod(artifact.shape)*args.G - 1
    k = int(k)*(k > 0)
    topk = torch.sort(artifact.view(-1), descending=True)[0][k]
    artifact = (artifact > topk).float()

    return artifact
