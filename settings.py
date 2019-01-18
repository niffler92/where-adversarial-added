import os
from pathlib import Path
from nsml import NSML_NFS_OUTPUT


if NSML_NFS_OUTPUT:
    PROJECT_ROOT = Path(os.path.join(NSML_NFS_OUTPUT, 'kyhoon', 'ACE-Defense'))
else:
    PROJECT_ROOT = Path(__file__).resolve().parent


MODEL_PATH_DICT = {
    'alexnet': 'alexnet-owt-4df8aa71.pth',
    'densenet121': 'densenet121-a639ec97.pth',
    'densenet161': 'densenet161-8d451a50.pth',
    'densenet169': 'densenet169-b2777c0a.pth',
    'densenet201': 'densenet201-c1103571.pth',
    'inception_v3': 'inception_v3_google-1a9a5a14.pth',
    'resnet101': 'resnet101-5d3b4d8f.pth',
    'resnet152': 'resnet152-b121ed2d.pth',
    'resnet18': 'resnet18-5c106cde.pth',
    'resnet34': 'resnet34-333f7ec4.pth',
    'resnet50': 'resnet50-19c8e357.pth',
    'squeezenet1_0': 'squeezenet1_0-a815701f.pth',
    'squeezenet1_1': 'squeezenet1_1-f364aa15.pth',
    'vgg11': 'vgg11-bbd30ac9.pth',
    'vgg11_bn': 'vgg11_bn-6002323d.pth',
    'vgg13_bn': 'vgg13_bn-abd245e5.pth',
    'vgg13': 'vgg13-c768596a.pth',
    'vgg16': 'vgg16-397923af.pth',
    'vgg16_bn': 'vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'vgg19_bn-c79401a0.pth',
    'vgg19': 'vgg19-dcbb9e9d.pth',
}
