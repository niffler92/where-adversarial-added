import logging
from collections import Iterable, OrderedDict
from pathlib import Path
import os
from datetime import datetime

from tensorboardX import SummaryWriter
from termcolor import colored
import torch
import numpy as np
#import seaborn as sns
from PIL import Image


LOG_LEVELS = {
    'DEBUG' : {'lvl': 10, 'color': 'white'},
    'INFO' : {'lvl': 20, 'color': 'green'},
    'WARNING' : {'lvl': 30, 'color': 'red'},
    'ERROR' : {'lvl': 40, 'color': 'red'},
    'CRITICAL' : {'lvl': 50, 'color': 'red'},
}

class Logger:
    def __init__(self, name, log_dir=None, verbose=1):
        logger = logging.getLogger(name)
        format = logging.Formatter("[%(name)s|%(levelname)s] %(asctime)s > %(message)s")
        streamHandler = logging.StreamHandler()
        streamHandler.setFormatter(format)
        logger.addHandler(streamHandler)
        logger.setLevel(verbose)

        if log_dir is not None:
            if not log_dir.exists():
                log_dir.mkdir(parents=True)
            log_dir = str(log_dir)
            fileHandler = logging.FileHandler(log_dir +
                                              '/{}.'.format(name) + datetime.now().strftime("%Y%m%d%H%M%S"),
                                              mode="w")
            fileHandler.setFormatter(format)
            logger.addHandler(fileHandler)

        self.log_dir = log_dir
        self.logger = logger
        self.t_logger = TLogger(log_dir)
        self.writer = self.t_logger.writer

    def log(self, msg, lvl="INFO"):
        """Print message to terminal"""
        lvl, color = self.get_level_color(lvl)
        self.logger.log(lvl, colored(msg, color))

    def add_level(self, name, lvl, color='white'):
        """Add logging level to environment"""
        global LOG_LEVELS
        if name not in LOG_LEVELS.keys() and lvl not in LOG_LEVELS.values():
            LOG_LEVELS[name] = {'lvl': lvl, 'color': color}
            logging.addLevelName(lvl, name)
        else:
            raise AssertionError("log level already exists")

    def scalar_summary(self, info, step, lvl="INFO"):
        assert isinstance(info, dict), "data must be a dictionary"
        msg = "Step: {:5d} |".format(step)
        for k in info.keys():
            v = info[k]
            msg += " {}: ".format(k)
            if isinstance(v, int):
                msg += "{:5d} |".format(v)
            elif isinstance(v, float):
                msg += "{:5.4f} |".format(v)
        msg = msg[:-2]

        self.log(msg, lvl)
        self.t_logger.scalar_summary(info, step)

    def image_summary(self, info, step, save=True):
        """Plot images to Tensorboard.
           The images are assumed to be of shape (C, W, H),
           with RGB values in range [0,1].
        """
        assert isinstance(info, dict), "data must be a dictionary"
        self.check_savedir()
        for k, img in info.items():
            assert len(img.shape) == 3, "image must be a 3D array (C, H, W)"
            if not isinstance(img, np.ndarray):
                try:
                    img = np.reshape(np.asarray(img), list(img.shape))
                    assert len(info[k]) == 3, "image must be a 3D array (C, H, W)"
                except:
                    raise TypeError("image must be an array-like instance")

            img = (img*255).astype(np.uint8)
            if img.shape[0] == 1:
                img = np.repeat(img, 3, axis=0)
            img = np.transpose(img, [1,2,0])
            info[k] = img

            if save:
                np.save(self.log_dir + '/numpy/' + k, img)
                if img.shape[-1] == 3:
                    img = Image.fromarray(img, 'RGB')
                else:
                    img = Image.fromarray(img, 'RGBA')
                img.save(self.log_dir + '/image/' + k + '.png')

        self.t_logger.image_summary(info, step)

    def check_savedir(self):
        if self.log_dir:
            if not os.path.exists(self.log_dir + '/numpy/'):
                os.makedirs(self.log_dir + '/numpy/')
            if not os.path.exists(self.log_dir + '/image/'):
                os.makedirs(self.log_dir + '/image/')

    def get_level_color(self, level):
        assert isinstance(level, str)

        global LOG_LEVELS
        level_num = LOG_LEVELS[level]['lvl']
        color = LOG_LEVELS[level]['color']

        return level_num, color


class TLogger:
    def __init__(self, log_dir=None):
        if log_dir is not None:
            self.writer = SummaryWriter(log_dir)
        else:
            self.writer = None

    def scalar_summary(self, info, step):
        if self.writer is not None:
            for k, v in info.items():
                self.writer.add_scalar(k, v, step)

    def image_summary(self, info, step):
        if self.writer is not None:
            for k, v in info.items():
                self.writer.add_image(k, v, step)
