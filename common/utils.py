import sys
sys.path.append("../")
from datetime import datetime
import os
import inspect
import numpy as np

from settings import PROJECT_ROOT


def get_dirname(args):
    path = os.path.join(PROJECT_ROOT.as_posix(), args.log_dir,
            datetime.now().strftime("%Y%m%d%H%M%S") + "-")
    path += "{}-".format(args.mode)
    path += "{}-".format(args.dataset)
    path += "{}".format(args.model)

    return path


def show_current_model(model, args):
    print("\n".join("{}: {}".format(k, v) for k, v in sorted(vars(args).items())))

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    total_params = np.sum([np.prod(p.size()) for p in model_parameters])

    print('%s\n\n'%(type(model)))
    print('%s\n\n'%(inspect.getsource(model.__init__)))
    print('%s\n\n'%(inspect.getsource(model.forward)))

    print("*"*40 + "%10s" % args.model + "*"*45)
    print("*"*40 + "PARAM INFO" + "*"*45)
    print("-"*95)
    print("| %40s | %25s | %20s |" % ("Param Name", "Shape", "Number of Params"))
    print("-"*95)
    for name, param in model.named_parameters():
        if param.requires_grad:
            print("| %40s | %25s | %20d |" % (name, list(param.size()), np.prod(param.size())))
    print("-"*95)
    print("Total Params: %d" % (total_params))
    print("*"*95)

