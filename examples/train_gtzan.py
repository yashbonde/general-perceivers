import math
import numpy as np
from sklearn import preprocessing
import librosa

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchaudio.datasets import GTZAN
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.utils.data import random_split

from scipy.signal import savgol_filter
import matplotlib.pyplot as plt

# -----
from gperc.utils import set_seed
from gperc import AudioConfig, Perceiver
from gperc.models import build_position_encoding

# -----

SAMPLE_RATE = 22050 # in Hertz
TRACK_DURATION = 30 # in seconds
SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION


class MyGtzanDataset(Dataset):
  def __init__(self, dataset, num_mfcc=13, n_fft=2048, hop_length=512, num_segments=5):
        super().__init__()
        samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
        num_mfcc_vectors_per_segment = math.ceil(samples_per_segment / hop_length)
        self.signals = [item[0].numpy().flatten() for item in dataset]
        self.sample_rates = [item[1] for item in dataset]
        self.signal_labels = [item[2] for item in dataset]
        le = preprocessing.LabelEncoder()
        le.fit(self.signal_labels)
        self.signal_labels = le.transform(self.signal_labels)
        self.mfcc=[]
        self.labels=[]
        for i in range(len(self.signals)):
          for segment in range(num_segments):
            start = samples_per_segment*segment
            finish = start+samples_per_segment
            mfcc = librosa.feature.mfcc(self.signals[i][start:finish], self.sample_rates[i], n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length).T
            if len(mfcc) == num_mfcc_vectors_per_segment:
              self.mfcc.append(mfcc.tolist())
              self.labels.append(self.signal_labels[i])
        self.mfcc=torch.Tensor(self.mfcc)
        self.labels=torch.LongTensor(self.labels)

  def __len__(self):
        return len(self.mfcc)

  def __getitem__(self, i):
    return self.mfcc[i], self.labels[i]


dataset = GTZAN(root=".", download=True)
data = MyGtzanDataset(dataset, num_segments=10)

test_size = 100
train_size = len(data) - test_size
train_ds, test_ds = random_split(data, [train_size, test_size])

batch_size = 32
dl_train = DataLoader(train_ds, batch_size, shuffle=True, drop_last=True)
dl_test = DataLoader(test_ds, batch_size, shuffle=False, drop_last=False)

config = AudioConfig(
    sample_rate=22050,
    duration=30,
    hop_length=512,
    num_mfcc=13,
    num_segments=10,
    num_channels=1,
    latent_len=32,
    latent_dim=32,
    num_layers=4,
    n_classes=10
)

class PerceiverGtzanClassifier(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.emb = build_position_encoding("trainable", config, config.input_len, config.input_dim)
        self.perceiver = Perceiver(config)

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
      x=torch.reshape(x,(x.shape[0],x.shape[1]*x.shape[2],1))
      pos_emb = torch.cat([self.emb[None, ...] for _ in range(x.shape[0])], dim=0)
      out = x + pos_emb
      return self.perceiver(out)

set_seed(config.seed)
model = PerceiverGtzanClassifier(config)
print("model parameters:", model.num_parameters())

iter_dl_train = iter(dl_train)
from tqdm import trange
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

plt.figure(figsize=(20,10))
plt.subplot(1, 2, 1)
plt.plot(savgol_filter(all_acc, window_length = 51, polyorder = 3))
plt.title("Training Accuracy")
plt.subplot(1, 2, 2)
plt.plot(all_loss)
plt.title("Training Loss")
plt.show()
