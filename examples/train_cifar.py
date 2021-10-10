from gperc import PerceiverConfig, Perceiver
from gperc.models import build_position_encoding

from torchvision.datasets import CIFAR10
from torchvision import transforms as TR
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import functional as F

from tqdm import trange

config = PerceiverConfig(
  input_len = 32 * 32,
  input_dim = 3,
  latent_len = 64,
  latent_dim = 64,
  num_layers = 12,
  ffw_latent = 16,
  output_len = 10,
  output_dim = 32,
  n_classes = 10,
  decoder_projection = True,
  dropout = 0.3,
)
print(config)

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

model = PerceiverCIFAR10(config)
print("model parameters:", model.num_parameters())

ds_train = CIFAR10('/tmp', train=True, download=True, transform=TR.Compose([
    TR.ToTensor(),
    TR.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    TR.Lambda(lambda x: x.reshape(-1, 3)),
  ])
)
ds_test = CIFAR10('/tmp', train=False, download=True, transform=TR.Compose([
    TR.ToTensor(),
    TR.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    TR.Lambda(lambda x: x.reshape(-1, 3)),
  ])
)

dl_train = DataLoader(ds_train, batch_size=128, shuffle=True, drop_last=True)
dl_test = DataLoader(ds_test, batch_size=128, shuffle=True, drop_last=True)

iter_dl_train = iter(dl_train)

pbar = trange(1000)
optim = Adam(model.parameters(), lr=4e-3)
all_loss = []; all_acc = []

for i in pbar:
  try:
    x, y = next(iter_dl_train)
  except StopIteration:
    iter_dl_train = iter(dl_train)
    x, y = next(iter_dl_train)
  
  _y = model(x)
  loss = F.cross_entropy(_y, y)
  all_loss.append(loss.item())
  all_acc.append((_y.argmax(dim=1) == y).sum().item() / len(y))
  pbar.set_description(f"loss: {loss.item():.4f} | acc: {all_acc[-1]:.4f}")
  loss.backward()
  torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
  optim.step()

  if i and i % 500 == 0:
    model.eval()
    with torch.no_grad():
      _all_loss = []; _all_acc = []
      for x, y in dl_test:
        _y = model(x)
        loss = F.cross_entropy(_y, y)
        _all_loss.append(loss.item())
        _all_acc.append(_y.argmax(-1) == y)
      print(f"Test Loss: {sum(_all_loss)} | Test Acc: {sum(_all_acc)/len(_all_acc)}")
    model.train()
