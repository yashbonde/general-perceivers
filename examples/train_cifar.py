from tempfile import gettempdir
from tqdm import trange
import numpy as np

import torch
from torch.optim import Adam
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms as TR
from torchvision.datasets import CIFAR10


from gperc import set_seed, build_position_encoding, ImageConfig, Perceiver



class PerceiverCIFAR10(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.emb = build_position_encoding("trainable", config, 1024, 3)
        self.perceiver = Perceiver(config)

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        pos_emb = torch.cat([self.emb[None, ...] for _ in range(x.shape[0])], dim=0)
        out = x + pos_emb
        return self.perceiver(out)


# define your datasets
ds_train = CIFAR10(
    gettempdir(),
    train=True,
    download=True,
    transform=TR.Compose(
        [
            TR.ToTensor(),
            TR.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            TR.Lambda(lambda x: x.permute(1, 2, 0).reshape(-1, 3)),
        ]
    ),
)
ds_test = CIFAR10(
    gettempdir(),
    train=False,
    download=True,
    transform=TR.Compose(
        [
            TR.ToTensor(),
            TR.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            TR.Lambda(lambda x: x.permute(1, 2, 0).reshape(-1, 3)),
        ]
    ),
)

# define the config and load the model
config = ImageConfig(
    image_shape=[32, 32, 3],
    latent_len=32,
    latent_dim=32,
    n_classes=10,
)

set_seed(config.seed)
model = PerceiverCIFAR10(config)
print("model parameters:", model.num_parameters())

# define the dataloaders, optimizers and lists
batch_size = 32
dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True, drop_last=True)
dl_test = DataLoader(ds_test, batch_size=batch_size, shuffle=False, drop_last=False)
iter_dl_train = iter(dl_train)

pbar = trange(10000)
optim = Adam(model.parameters(), lr=0.001)
all_loss = []
all_acc = []

# train!
for i in pbar:
    try:
        x, y = next(iter_dl_train)
    except StopIteration:
        iter_dl_train = iter(dl_train)
        x, y = next(iter_dl_train)

    optim.zero_grad()
    _y = model(x)
    loss = F.cross_entropy(_y, y)
    all_loss.append(loss.item())
    all_acc.append((_y.argmax(dim=1) == y).sum().item() / len(y))
    pbar.set_description(f"loss: {np.mean(all_loss[-50:]):.4f} | acc: {all_acc[-1]:.4f}")
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optim.step()

    if (i + 1) % 500 == 0:
        model.eval()
        with torch.no_grad():
            _all_loss = []
            _all_acc = []
            for x, y in dl_test:
                _y = model(x)
                loss = F.cross_entropy(_y, y)
                _all_loss.append(loss.item())
                _all_acc.append((_y.argmax(-1) == y).sum().item() / len(y))
            print(f"Test Loss: {sum(_all_loss)} | Test Acc: {sum(_all_acc)/len(_all_acc)}")
        model.train()
