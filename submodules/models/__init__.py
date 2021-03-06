import sys
import os
sys.path.append("../../")
import settings
sys.path.append(os.path.join(settings.PROJECT_ROOT.as_posix(), 'submodules/models'))

from torchvision.models import *
"""torchvision models are lower case models.
They are One of the followings:
    [alexnet, densenet, densenet121, densenet161, densenet169, densenet201, inception, inception_v3,
    resnet, resnet101, resnet152, resnet18, resenet34, resnet50, squeezenet, squeezenet1_0, squeezenet1_1,
    vgg, vgg11, vgg11_bn, vgg13, vgg13_bn, vgg16, vgg16_bn, vgg19, vgg19_bn]
"""

from vgg import *
from lenet import *
from resnet import *
from submodules.enhancer import *
from submodules.autoencoders import *
