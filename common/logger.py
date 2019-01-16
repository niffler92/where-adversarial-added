import logging
import os
from datetime import datetime

from termcolor import colored
import numpy as np

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
            os.makedirs(log_dir, exist_ok=True)
            fileHandler = logging.FileHandler(log_dir +
                                              '/{}.'.format(name) + datetime.now().strftime("%Y%m%d%H%M%S"),
                                              mode="w")
            fileHandler.setFormatter(format)
            logger.addHandler(fileHandler)

        self.log_dir = log_dir
        self.logger = logger

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

    def get_level_color(self, level):
        assert isinstance(level, str)
        global LOG_LEVELS
        level_num = LOG_LEVELS[level]['lvl']
        color = LOG_LEVELS[level]['color']

        return level_num, color

