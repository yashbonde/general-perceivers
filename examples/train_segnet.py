import os
from glob import glob
from tempfile import gettempdir
from tqdm import trange
import numpy as np
from PIL import Image
from collections import Counter

import subprocess

import torch
from torch.optim import Adam
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as TR

# -----
from gperc.utils import set_seed
from gperc import ImageConfig, Perceiver
from gperc.models import build_position_encoding

# -----

DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"
print("DEVICE:", DEVICE)


class MySegnetData(Dataset):
    def __init__(self, root_folder = "./segnet-images/"):
        if not os.path.exists(root_folder):
            
            images = "https://www.robots.ox.ac.uk/~vgg/data/iseg/data/images.tgz"
            image_gt = "https://www.robots.ox.ac.uk/~vgg/data/iseg/data/images-gt.tgz"
            
            subprocess.run(["wget", images, "-O", "./images.tgz"])
            subprocess.run(["tar", "-xf", "./images.tgz"])
            subprocess.run(["mv", "./images/", root_folder])
            os.remove("./images.tgz")

            subprocess.run(["wget", image_gt, "-O", "./images-gt.tgz"])
            subprocess.run(["tar", "-xf", "./images-gt.tgz"])
            subprocess.run(["mv", "./images-gt/", root_folder])
            os.remove("./images-gt.tgz")
        
        images = sorted(glob(os.path.join(root_folder, "*.jpg")))
        masks = sorted(glob(os.path.join(root_folder, "images-gt/*.png")))

        self.ds = []
        for i in images:
            idx_ = i.split("/")[-1].split(".")[0]
            if os.path.join(root_folder, f"images-gt/{idx_}.png") in masks:
                self.ds.append([i, os.path.join(root_folder, f"images-gt/{idx_}.png")])

        self.transform = TR.Compose(
            [
                TR.ToTensor(),
                TR.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                TR.Lambda(lambda x: x.permute(1, 2, 0).reshape(-1, 3)),
            ]
        )

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        src_img, trg_img = self.ds[i]
        src = Image.open(src_img).resize((224, 224))
        trg = Image.open(trg_img).resize((224, 224))

        src = self.transform(src)
        trg = torch.from_numpy(np.array(trg).astype(np.int32)).long()
        trg[trg > 0] = 1.
        return src, trg

        
ds_train = MySegnetData()
print(ds_train[0])

config = ImageConfig(
    image_shape = (224, 224, 3),
    latent_len = 32,
    latent_dim = 32,
    num_layers = 6,
    n_classes = 2,
    task = "segmentation",
)
print(config)

class PerceiverSegnet(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.emb = build_position_encoding("trainable", config, config.input_len, 3)
        self.perceiver = Perceiver(config)

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x, return_attentions = False):
        pos_emb = torch.cat([self.emb[None, ...] for _ in range(x.shape[0])], dim=0)
        out = x + pos_emb
        out = self.perceiver(out, return_attentions=return_attentions)
        return out

set_seed(config.seed)
model = PerceiverSegnet(config)
print("model parameters:", model.num_parameters())

if DEVICE != "cpu":
    print("Moving to GPU")
    model = model.to(DEVICE)


# define the dataloaders, optimizers and lists
batch_size = 32
dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True, drop_last=True)
iter_dl_train = iter(dl_train)

pbar = trange(10000)
optim = Adam(model.parameters(), lr=0.001)
all_loss = []
all_acc = []

# train!
# accuracy for this case is not the best kind of metric because large percentage of the data is 0
for i in pbar:
    try:
        x, y = next(iter_dl_train)
    except StopIteration:
        iter_dl_train = iter(dl_train)
        x, y = next(iter_dl_train)

    optim.zero_grad()
    _y = model(x.to(DEVICE)).cpu()
    _y = _y.contiguous().view(-1, _y.shape[-1])
    y = y.contiguous().view(-1)
    loss = F.cross_entropy(_y, y)
    all_loss.append(loss.item())
    all_acc.append((_y.argmax(dim=-1) == y).sum().item() / len(y))
    pbar.set_description(f"loss: {np.mean(all_loss[-50:]):.4f} | acc: {all_acc[-1]:.4f}")
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optim.step()
