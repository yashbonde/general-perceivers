import os
import time
import torch
import numpy as np
import random


def set_seed(s=4):
    """set seed for controlling randomness"""
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_files_in_folder(folder, ext=[".txt"], sort=True):
    # this method is faster than glob
    all_paths = []
    for root, _, files in os.walk(folder):
        for f in files:
            for e in ext:
                if f.endswith(e):
                    all_paths.append(os.path.join(root, f))
    return sorted(all_paths) if sort else all_paths

def folder(x):
    # get the folder of this file path
    return os.path.split(os.path.abspath(x))[0]

def join(x, *args):
    return os.path.join(x, *args)

def timeit(fn):
  def _fn(*args, **kwargs):
    start = time.time()
    out = fn(*args, **kwargs)
    return time.time() - start, out
  return _fn

