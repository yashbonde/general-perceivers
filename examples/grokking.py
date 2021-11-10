#!/usr/bin/env python3
"""
Grokking
========

The phenomenon of model snapping into place and learn the rule as you keep on training.
"""

import random
import numpy as np
from tqdm import trange
from types import SimpleNamespace

import torch
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader

from gperc import Perceiver, PerceiverConfig
from gperc.models import build_position_encoding


def create_dataset(func, split_perc, modulo=97, verbose=False):
    if modulo > 10000:
        raise ValueError("modulo too large, code will be very slow. Please re write vectorised numpy code")

    # ops: loaded by eval later
    _plus_op = lambda x, y, m: (x + y) % m
    _minus_op = lambda x, y, m: (x - y) % m
    _div_op = lambda x, y, m: (x / y) % m
    _odd_even_op = lambda x, y, m: (x / y) % m if y % 2 else (x - y) % m
    _square_op = lambda x, y, m: (x ** 2 + y ** 2) % m
    _sum_square_op = lambda x, y, m: (x ** 2 + x * y + y ** 2) % m

    func = eval(func)

    # create the dataset
    numbers = list(range(modulo))
    dataset = []
    for x in numbers:
        for y in numbers:
            dataset.append((x, y, func(x, y, modulo)))

    # print if needed
    if verbose:
        for d in dataset:
            print(d)

    # split the dataset into train/test and create datasets
    train_dataset = []
    test_dataset = []
    for d in dataset:
        if random.uniform(0, 1) > split_perc:
            test_dataset.append(d)
        else:
            train_dataset.append(d)

    return train_dataset, test_dataset


modulo = 30  # this is also the embedding size
verbose = False
split_perc = 0.9
op_name = "_plus_op"

train_dataset, test_dataset = create_dataset(op_name, split_perc, modulo, verbose)
train_dataset = torch.tensor(train_dataset).long()
train_dataset = train_dataset[torch.randperm(train_dataset.size()[0]), :]  # row shuffle
test_dataset = torch.tensor(test_dataset).long()  # no need to shuffle
print("train_dataset:", train_dataset.shape)
print("test_dataset:", test_dataset.shape)


class GrokkingConfig(PerceiverConfig):
    def __init__(
        self,
        latent_dim: int,
        modulo: int,
        max_len: int,
        latent_frac=1,
    ):
        super().__init__()
        self.vocab_size = modulo
        self.max_len = max_len

        self.input_len = max_len
        self.input_dim = latent_dim
        self.latent_len = int(latent_frac * max_len)
        self.latent_dim = latent_dim
        self.output_len = max_len
        self.output_dim = latent_dim

        self.decoder_cross_attention = False
        self.decoder_residual = False
        self.decoder_projection = True
        self.n_classes = modulo


class GrokkingPerceiver(torch.nn.Module):
    def __init__(self, config: GrokkingConfig):
        super().__init__()
        self.config = config
        self.emb = torch.nn.Embedding(config.vocab_size, config.input_dim)
        self.pos = build_position_encoding("trainable", config, config.max_len, config.latent_dim)
        self.perceiver = Perceiver(config)

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        pos = torch.cat([self.pos[None, ...] for _ in range(x.shape[0])], dim=0)
        x = self.emb(x) + pos
        out = self.perceiver(x)
        return out


config = GrokkingConfig(latent_dim=32, modulo=modulo, max_len=test_dataset.shape[1] - 1, latent_frac=1.0)
model = GrokkingPerceiver(config)
print("model params:", model.num_parameters())
train_config = SimpleNamespace(
    num_steps=int(1e4),
    batch_size=528,
    epochs=100,
    lr=3e-4,
    weight_decay=0,
)

# perform shape validation
if verbose:
    with torch.no_grad():
        _x = test_dataset[:100, :2]
        out = model(_x)
        print(" in:", _x.shape)
        print("out:", out.shape)
        assert out.shape == (_x.shape[0], config.n_classes)

# train
pbar = trange(train_config.num_steps)
optim = torch.optim.AdamW(model.parameters(), lr=train_config.lr, weight_decay=train_config.weight_decay)
all_loss = []
all_acc = []
train_dataset = DataLoader(TensorDataset(train_dataset), batch_size=train_config.batch_size, shuffle=True)
test_dataset = DataLoader(TensorDataset(test_dataset), batch_size=train_config.batch_size, shuffle=True)
_iter_train_dataset = iter(train_dataset)

for i in pbar:
    try:
        x = next(_iter_train_dataset)[0]
    except StopIteration:
        _iter_train_dataset = iter(train_dataset)
        x = next(_iter_train_dataset)[0]

    optim.zero_grad()
    _y = model(x[:, :-1])
    _y = _y.contiguous().view(-1, _y.shape[-1])
    tensor_target = x[:, -1:].contiguous().view(-1)
    # print(_y.argmax(dim=-1), tensor_target)
    loss = F.cross_entropy(_y, tensor_target)

    all_loss.append(loss.item())
    all_acc.append((_y.argmax(dim=1) == tensor_target).sum().item() / len(tensor_target))
    loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optim.step()

    if i % 500 == 0:
        print(_y.argmax(dim=1)[:100])
        print(tensor_target[:100])

    # now to the eval loop
    with torch.no_grad():
        all_loss_test = []
        all_acc_test = []
        for x in test_dataset:
            x = x[0]
            out = model(x[:, :-1])
            out = out.contiguous().view(-1, out.shape[-1])
            tensor_target = x[:, -1:].contiguous().view(-1)
            loss_test = F.cross_entropy(out, tensor_target)
            all_loss_test.append(loss_test.item())
            all_acc_test.append((out.argmax(dim=1) == tensor_target).sum().item() / len(tensor_target))

    pbar.set_description(
        f"loss: {np.mean(all_loss[-50:]):.4f} | acc: {all_acc[-1]:.4f} | loss_test: {np.mean(all_loss_test):.4f} | acc_test: {np.mean(all_acc_test):.4f}"
    )
