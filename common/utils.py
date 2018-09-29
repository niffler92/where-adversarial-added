import sys
sys.path.append("../")
from pathlib import Path
from datetime import datetime
import contextlib
import logging
import getpass
import uuid
import json
import copy
import os

from common.logger import Logger
import settings
from nsml import DATASET_PATH, HAS_DATASET


def update_train_dir(args):
    username = getpass.getuser()
    timestamp = datetime.now().strftime("%y%m%d%H%M%S")
    if hasattr(args, "train_dir") and username not in args.train_dir:
        # XXX using username as condition for making subdir
        subdir = Path("{}-{}-".format(username, timestamp) +
                      "{}-".format(uuid.uuid1().hex[:6]) +
                      "{}".format(args.tag))
        args.train_dir = (Path(args.train_dir) / subdir).as_posix()
        log.info(args.train_dir)


def dump_configuration(args):
    if hasattr(args, "train_dir") and not Path(args.train_dir).exists():
        Path(args.train_dir).mkdir(parents=True)
        directory = Path(args.train_dir)
    else:
        directory = Path(settings.PROJECT_ROOT) / Path("data/working/")

    payload = copy.deepcopy(vars(args))

    # To prevent TypeError: Object of type 'function' is not JSON serializable
    for k, v in vars(args).items():
        is_pop = False

        if callable(v):
            is_pop = True

        if is_pop:
            payload.pop(k)
            log.info("pop {}".format(k))

    json.dump(payload, open(directory / Path("config.json"), "w"))


@contextlib.contextmanager
def timer(name):
    hf_timer = hf.Timer()
    yield
    log.info("<Timer> {} : {}".format(name, hf_timer.rounded))


def timeit(method):
    def timed(*args, **kw):
        hf_timer = hf.Timer()
        result = method(*args, **kw)
        log.info("<Timeit> {!r} ({!r}, {!r}) {}".format(method.__name__, args, kw, hf_timer.rounded))
        return result
    return timed


def find_class_by_name(name, modules):
    """Searches the provided modules for the named class and returns it."""
    modules = [getattr(module, name, None) for module in modules]
    return next(a for a in modules if a)


def get_checkpoint(args=None):
    if args.ckpt_name:
        if DATASET_PATH:
            root = os.path.join(DATASET_PATH, 'train', 'checkpoints', 'checkpoints_ace', args.ckpt_name)
            if HAS_DATASET and os.path.isfile(root):
                print("Running on NSML")
                return root
        root = os.path.join('checkpoints', args.ckpt_name)
    else:
        if DATASET_PATH:
            root = os.path.join(DATASET_PATH, 'train', 'checkpoints', 'checkpoints_ace', args.model)
            if HAS_DATASET and os.path.isfile(root):
                print("Running on NSML")
                return root
        root = os.path.join('checkpoints', args.dataset, args.model)
    assert os.path.isfile(root), "Checkpoint file does not exist."
    print("Running on Local")
    return root
