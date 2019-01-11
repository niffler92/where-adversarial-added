import logging
import os
from datetime import datetime

import nsml
import visdom
from tensorboardX import SummaryWriter
from termcolor import colored
import torch
from torch.autograd import Variable

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
            log_dir.mkdir(parents=True, exist_ok=True)
            log_dir = str(log_dir)
            fileHandler = logging.FileHandler(log_dir +
                                              '/{}.'.format(name) + datetime.now().strftime("%Y%m%d%H%M%S"),
                                              mode="w")
            fileHandler.setFormatter(format)
            logger.addHandler(fileHandler)

        self.log_dir = log_dir
        self.logger = logger
        self.n_logger = NLogger()
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
        self.n_logger.scalar_summary(info, step)
        self.t_logger.scalar_summary(info, step)

    def image_summary(self, info, step, save_dir=None):
        """Plot images to NSML and Tensorboard.
           The images are assumed to be of shape (C, W, H),
           with RGB values in range [0,1].
        """
        assert isinstance(info, dict), "data must be a dictionary"
        assert save_dir, "no save_dir"

        self.check_savedir(save_dir)
        for k, img in info.items():
            if isinstance(img, Variable):
                img = img.data
            img = img.cpu().numpy()
            assert len(img.shape) == 3, "image must be a 3D array (C, H, W)"
            np.save(os.path.join(self.log_dir, save_dir, 'numpy', str(k)), img)

            # Save as PNG file
            img = np.transpose(img, [1, 2, 0])
            if img.shape[-1] == 3:
                img = Image.fromarray(img, 'RGB')
            else:
                img = Image.fromarray(img, 'RGBA')
            img.save(os.path.join(self.log_dir, save_dir, 'image', str(k) + '.png'))

        # FIXME: temporarily remove image_summary
        # self.n_logger.image_summary(info, step)
        # self.t_logger.image_summary(info, step)

    def histo_summary(self, info, step):
        raise NotImplementedError

    def heatmap_summary(self, info, step, save=True):
        assert isinstance(info, dict), "data must be a dictionary"
        for k, img in info.items():
            assert len(img.shape) == 2, "image must be a 2D array"
            if not isinstance(img, np.ndarray):
                try:
                    img = np.reshape(np.asarray(img), list(img.shape))
                    info[k] = img
                except:
                    raise TypeError("image must be an array-like instance")
        self.n_logger.heatmap_summary(info, step)

        # FIXME NSML can't import seaborn... even with setup.py
        if save:
            self.check_savedir()
            for k, v in info.items():
                np.save(self.log_dir + '/numpy/' + k, v)

        #        heatmap = sns.heatmap(v)
        #        heatmap.figure.savefig(
        #                fname="{}/image/{}.png".format(self.log_dir, k)
        #                )

    def check_savedir(self, save_dir=None):
        if self.log_dir:
            if save_dir is not None:
                log_dir = os.path.join(self.log_dir, save_dir)
            else:
                log_dir = self.log_dir

            if not os.path.exists(log_dir + '/numpy/'):
                os.makedirs(log_dir + '/numpy/')
            if not os.path.exists(log_dir + '/image/'):
                os.makedirs(log_dir + '/image/')

    def get_level_color(self, level):
        assert isinstance(level, str)

        global LOG_LEVELS
        level_num = LOG_LEVELS[level]['lvl']
        color = LOG_LEVELS[level]['color']

        return level_num, color


class NLogger:
    def __init__(self):
        if nsml.IS_ON_NSML:
            self.viz = nsml.Visdom(visdom=visdom, use_incoming_socket=False)

    def scalar_summary(self, info, step):
        if nsml.IS_ON_NSML:
            nsml.report(step=step, **info)

    def image_summary(self, info, step):
        if nsml.IS_ON_NSML:
            for k, v in info.items():
                self.viz.image(v, opts=dict(title='{}/{}'.format(k, step),
                                            caption='{}/{}'.format(k, step)))

    def histo_summary(self, info, step):
        raise NotImplementedError

    def heatmap_summary(self, info, step):
        if nsml.IS_ON_NSML:
            for k, v in info.items():
                if isinstance(v, torch.Tensor) or isinstance(v, torch.cuda.FloatTensor):  # XXX Hmm didn't have this error before
                    v = v.cpu().numpy()

                self.viz.heatmap(v, opts=dict(title='{}/{}'.format(k, step),
                                              caption='{}/{}'.format(k, step),
                                              rownames=list(range(v.shape[0])),
                                              columnnames=list(range(v.shape[1])),
                                              colormap='Electric'))
                #self.viz.heatmap(v, opts=dict(title='{}/{}'.format(k, step),
                #                              caption='{}/{}'.format(k, step),
                #                              rownames=list(range(v.size(0))),
                #                              columnnames=list(range(v.size(1))),
                #                              colormap='Electric'))


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
                v = np.rollaxis(v,0,3)
                v = np.rollaxis(v,0,-1)
                self.writer.add_image(k, v, step)

    def histo_summary(self, info, step):
        raise NotImplementedError

    def heatmap_summary(self, info, step):
        raise NotImplementedError
