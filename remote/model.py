import os
import random
import numpy as np

import torch
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader

from gperc import Perceiver, PerceiverConfig
from gperc.models import build_position_encoding


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

        self.modulo = modulo


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


class GrokkingTrainer:
    def __init__(
        self,
        model_id: str,
        model: GrokkingPerceiver,
        func: str = "_plus_op",
        batch_size: int = 32,
        lr: float = 3e-4,
        weight_decay: float = 0.001,
        num_steps: int = int(1e4),
        split_perc: float = 0.9,
        **kwargs,
    ):
        self.model_id = model_id
        self.model = model
        self.func = func
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.num_steps = num_steps
        self.split_perc = split_perc

        self.pid = os.getpid()

        for k, v in kwargs.items():
            setattr(self, k, v)

        _plus_op = lambda x, y, m: (x + y) % m
        _minus_op = lambda x, y, m: (x - y) % m
        _div_op = lambda x, y, m: (x / y) % m
        _odd_even_op = lambda x, y, m: (x / y) % m if y % 2 else (x - y) % m
        _square_op = lambda x, y, m: (x ** 2 + y ** 2) % m
        _sum_square_op = lambda x, y, m: (x ** 2 + x * y + y ** 2) % m

        self.func = eval(func)

        # create the dataset
        numbers = list(range(self.model.config.modulo))
        dataset = []
        for x in numbers:
            for y in numbers:
                dataset.append((x, y, self.func(x, y, self.model.config.modulo)))

        # # print if needed
        # if verbose:
        #   for d in dataset:
        #     print(d)

        # split the dataset into train/test and create datasets
        train_dataset = []
        test_dataset = []
        for d in dataset:
            if random.uniform(0, 1) > self.split_perc:
                test_dataset.append(d)
            else:
                train_dataset.append(d)

        train_dataset = torch.tensor(train_dataset).long()
        test_dataset = torch.tensor(test_dataset).long()
        print(f">> [{self.pid}] train_dataset:", train_dataset.shape)
        print(f">> [{self.pid}] test_dataset:", test_dataset.shape)
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

    def train(self, verbose=True):
        print(f">> [{self.pid}] starting training for model_id: {self.model_id}")

        # train
        self.model.train()
        pbar = range(self.num_steps)
        optim = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        all_loss = []
        all_acc = []
        train_dataset = DataLoader(TensorDataset(self.train_dataset), batch_size=self.batch_size, shuffle=True)
        test_dataset = DataLoader(TensorDataset(self.test_dataset), batch_size=self.batch_size, shuffle=True)
        _iter_train_dataset = iter(train_dataset)

        for i in pbar:
            try:
                x = next(_iter_train_dataset)[0]
            except StopIteration:
                _iter_train_dataset = iter(train_dataset)
                x = next(_iter_train_dataset)[0]

            optim.zero_grad()
            _y = self.model(x[:, :-1])
            _y = _y.contiguous().view(-1, _y.shape[-1])
            tensor_target = x[:, -1:].contiguous().view(-1)
            # print(_y.argmax(dim=-1), tensor_target)
            loss = F.cross_entropy(_y, tensor_target)

            all_loss.append(loss.item())
            all_acc.append((_y.argmax(dim=1) == tensor_target).sum().item() / len(tensor_target))
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optim.step()

            # now to the eval loop
            with torch.no_grad():
                self.model.eval()
                all_loss_test = []
                all_acc_test = []
                for x in test_dataset:
                    x = x[0]
                    out = self.model(x[:, :-1])
                    out = out.contiguous().view(-1, out.shape[-1])
                    tensor_target = x[:, -1:].contiguous().view(-1)
                    loss_test = F.cross_entropy(out, tensor_target)
                    all_loss_test.append(loss_test.item())
                    all_acc_test.append((out.argmax(dim=1) == tensor_target).sum().item() / len(tensor_target))
                self.model.train()

        print(
            f">> [{self.model_id}] loss: {np.mean(all_loss[-50:]):.4f} | acc: {all_acc[-1]:.4f} | loss_test: {np.mean(all_loss_test):.4f} | acc_test: {np.mean(all_acc_test):.4f}"
        )
        print(">> finished process for model_id:", self.model_id)
        return all_loss, all_acc, all_loss_test, all_acc_test
