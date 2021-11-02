# this script demonstrates use of gperc.Consumer object

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

import random
import numpy as np
from tqdm import trange
from nbox.utils import folder, join
from gperc import PerceiverMLM, TextConfig, Consumer, set_seed, get_files_in_folder
from gperc.data import get_vocab

# this method tests the Consumer on itself, going to docs
folder_path = join(folder(folder(__file__)), "docs", "source")
labels = ["tinker", "tailor", "soldier", "spy"]  # some labels
dataset = {}
for i, f in enumerate(get_files_in_folder(folder_path, [".rst"])):
    dataset[f] = labels[i % len(labels)]

# create the dataset
data = Consumer(dataset, n_bytes=1, seqlen=128, verbose=True, class_to_id={"tinker": 0, "tailor": 1, "soldier": 2, "spy": 3})
data.set_unsupervised_mode()
out = data[[1, 2, 3, 4], "unsupervised"]
print("Sample data:", {k: v.shape for k, v in out.items()})
vocab = get_vocab(1)
print(data)
print("--------- Creating batches ...")

# create the model
config = TextConfig(latent_dim=8, vocab_size=len(vocab), max_len=128, latent_frac=1.0)
set_seed(config.seed)

data.create_batches(batch_size=32, drop_last=True, seed=config.seed)
print(data)
sample = data.get_next_batch("unsupervised")
print({k: v.shape for k, v in sample.items()})

model = PerceiverMLM(config)
print("Number of parameters in the model:", model.num_parameters())

# pass data through the model
sample = data[[1, 2, 3, 4]]
with torch.no_grad():
    out = model(sample["input_array"])
print("Forward pass shape:", out.shape)

# train the model - unsupervised
pbar = trange(10)
optim = torch.optim.Adam(model.parameters(), lr=3e-4)
all_loss = []
all_acc = []

for i in pbar:
    batch = data.get_next_batch("unsupervised")

    optim.zero_grad()
    _y = model(batch["input_array"])
    _y = _y.contiguous().view(-1, _y.shape[-1])
    tensor_target = batch["output_tensor"].contiguous().view(-1)
    loss = F.cross_entropy(_y, tensor_target)

    all_loss.append(loss.item())
    all_acc.append((_y.argmax(dim=1) == tensor_target).sum().item() / len(tensor_target))
    pbar.set_description(f"loss: {np.mean(all_loss[-50:]):.4f} | acc: {all_acc[-1]:.4f}")
    loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optim.step()


# train the model - supervised
data.set_supervised_mode()
pbar = trange(4000)
optim = torch.optim.Adam(model.parameters(), lr=1e-5)
all_loss = []
all_acc = []

for i in pbar:
    batch = data.get_next_batch("supervised")

    print({k: v.shape for k, v in batch.items()})

    optim.zero_grad()
    _y = model(batch["input_array"])
    print("!@#$!@#$@!#$!@#$23", _y.shape)
    _y = _y.contiguous().view(-1, _y.shape[-1])
    loss = F.cross_entropy(_y, batch["class"])

    all_loss.append(loss.item())
    all_acc.append((_y.argmax(dim=1) == tensor_target).sum().item() / len(tensor_target))
    pbar.set_description(f"loss: {np.mean(all_loss[-50:]):.4f} | acc: {all_acc[-1]:.4f}")
    loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optim.step()
