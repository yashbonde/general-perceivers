from tempfile import gettempdir
from tqdm import trange
import numpy as np

import torch
from torch.optim import Adam
from torch.nn import functional as F
from torch.utils.data import DataLoader

# -----
from gperc.utils import set_seed
from gperc import ImageConfig, Perceiver
from gperc.models import build_position_encoding
# -----

# create a dummy random image and mask
x = torch.randn(100, 224*224, 3)
y = torch.randint(0, 10, (100, 224*224))

config = ImageConfig(
    image_shape = (224, 224, 3),
    latent_len = 32,
    latent_dim = 32,
    num_layers = 6,
    n_classes = 10,
    task = "segmentation",
)

print(config)
