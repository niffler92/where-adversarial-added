import sys
import os
from settings import PROJECT_ROOT
sys.path.append(os.path.join(PROJECT_ROOT.as_posix(), 'submodules/autoencoders'))

from unet import *
