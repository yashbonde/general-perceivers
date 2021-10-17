from tempfile import gettempdir
from tqdm import trange
import numpy as np

import torch
from torch.optim import Adam
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms as TR
from torchvision.datasets import SBDataset

# -----
from gperc.utils import set_seed
from gperc import ImageConfig, Perceiver
from gperc.models import build_position_encoding
# -----


ds_train = SBDataset(
    root=gettempdir(),
    image_set='',
    download=True
)
print(ds_train[0])



