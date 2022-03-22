# testing to see if training on CSV and then using CSV like format to query the model

import os
import json
import requests
import numpy as np
import pandas as pd
from tempfile import gettempdir

import torch
from torch.nn import functional
from tqdm.std import trange
import torch_optimizer as toptim

from gperc import TextConfig, Perceiver

def pre():

  fp = gettempdir() + "/titanic.csv"
  if not os.path.exists(fp):
    r = requests.get("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")
    r.raise_for_status()
    with open(fp, "wb") as f:
      f.write(r.content)

  df = pd.read_csv(fp)
  df = df.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1)

  # though the keys are the in the the following order, "Survived" is the target
  # and is easy because 0th index
  # ['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
  samples = []
  for x in json.loads(df.to_json(orient="records")):
    x["Sex"] = 0 if x["Sex"] == "male" else 1
    samples.append("|".join([str(y) for y in x.values()]))

  vocab = {k:i for i,k in enumerate(
    sorted(
      list(
        set("".join(samples))
        ) + ["?"]
      )
    )
  }
  maxlen = max([len(x) for x in samples])
  tensor = torch.zeros(len(samples), maxlen).long()
  attention_mask = torch.zeros(len(samples), maxlen).long()
  for i,s in enumerate(samples):
    tensor[i][:len(s)] = torch.tensor([vocab[c] for c in s])
    attention_mask[i][:len(s)] = 1

  print("===== Tensor")
  print(tensor.shape)
  torch.random.manual_seed(420)
  tensor = tensor[torch.randperm(len(tensor))]
  print(tensor)

  train = tensor[:int(len(tensor)*0.8)]
  train_att = attention_mask[:int(len(tensor)*0.8)]
  test = tensor[int(len(tensor)*0.8):]
  test_att = attention_mask[int(len(tensor)*0.8):]

  # create the model
  config = TextConfig(
    latent_dim = len(vocab) // 2,
    vocab_size = len(vocab),
    max_len = tensor.shape[1],
    latent_frac=0.25,
    ffw_ratio=1.0,
    num_layers = 6,
    num_heads = 1
  )
  model = Perceiver(config)
  print(model.num_parameters())

  return train, train_att, test, test_att, model, vocab

def main(n = 1000, lr = 3e-4, p = 0.85, optim = "Adam"):
  _torch = hasattr(torch.optim, optim)
  _toptim = hasattr(toptim, optim)
  if not _torch and not _toptim:
    raise ValueError("Unknown optimizer {}".format(optim))

  train, train_att, test, test_att, model, vocab = pre()
  target_train = train.clone().contiguous().view(-1)
  target_test = test.clone().contiguous().view(-1)

  optimizer = getattr(torch.optim, optim) if _torch else getattr(toptim, optim)
  optimizer = optimizer(model.parameters(), lr=lr)

  pbar = trange(n)
  for i in pbar:
    model.train()
    _this_sample = train.clone()
    mask = np.random.uniform(0, 1, _this_sample.shape) > p
    _this_sample[mask] = vocab["?"]
    _this_sample[:, 0] = vocab["?"]

    out_train = model(train, train_att)
    out_train = out_train.contiguous().view(-1, out_train.shape[-1])
    loss_train = functional.cross_entropy(out_train, target_train)
    acc_ = out_train.argmax(dim=-1) == target_train
    acc_class = acc_[mask.reshape(-1)].sum().item() / mask.sum().item()
    acc_avg = acc_.sum().item() / len(acc_)

    model.zero_grad()
    optimizer.zero_grad()
    loss_train.backward()
    optimizer.step()

    with torch.no_grad():
      model.eval()
      _this_sample = test.clone()
      mask = np.random.uniform(0, 1, _this_sample.shape) > p
      _this_sample[mask] = vocab["?"]
      _this_sample[:, 0] = vocab["?"]

      out_test = model(test, test_att)
      out_test = out_test.contiguous().view(-1, out_test.shape[-1])
      loss_test = functional.cross_entropy(out_test, target_test)
      acc_ = out_test.argmax(dim=-1) == target_test
      acc_class_test = acc_[mask.reshape(-1)].sum().item() / mask.sum().item()
      acc_avg_test = acc_.sum().item() / len(acc_)

    pbar.set_description(
      f"[{i:05d}/{n:05d} {i/n:0.3f}] "
      f"[Train] loss: {loss_train.item():.4f} acc: {acc_avg:.4f} acc_surv: {acc_class:.4f} "
      f"[Test] loss: {loss_test.item():.4f} acc: {acc_avg_test:.4f} acc_surv: {acc_class_test:.4f}"
    )

if __name__ == "__main__":
  from fire import Fire
  Fire(main)
