# Script to train a gperc model on time series data from NSE Stock dataset for TATAMOTORS
# Dataset preparation from https://github.com/ElisonSherton/TimeSeriesForecastingNN/blob/main/curateData.py

from nsepy import get_history
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from tqdm import trange

import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam

from gperc.utils import set_seed
from gperc import PerceiverConfig, Perceiver
from gperc.models import build_position_encoding

from scipy.signal import savgol_filter
import matplotlib.pyplot as plt

# --------------- Preparing the dataset --------------- #
def get_dataset(symbol,start_date,end_date):
    df = get_history(symbol=symbol,start=start_date,end=end_date)
    df['Date']=df.index
    return df


def prepare_data(df,price_column,date_column,num_steps=5):
  for i in range(num_steps):
    df[f"lag_{i + 1}"] = df[price_column].shift(periods = (i + 1))
  new_df = df[[date_column, price_column] + [f"lag_{x + 1}" for x in range(num_steps)]]
  new_df = new_df.iloc[num_steps:-1, :]

  inputs,outputs = [],[]
  for record in new_df.itertuples():
    input  = record[-num_steps:][::-1]
    output = record[-(num_steps+1)]
    inputs.append(input)
    outputs.append(output)

  size = len(inputs)
  inputs  = np.array(inputs)
  outputs = np.array(outputs)
  transformation = MinMaxScaler(feature_range=(-1, 1))
  inputs  = transformation.fit_transform(inputs)
  outputs = transformation.fit_transform(outputs.reshape(-1, 1))
  trainX = inputs[:int(0.8 * size)]
  trainY = outputs[:int(0.8 * size)]
  testX  = inputs[int(0.2 * size):]
  testY  = outputs[int(0.2 * size):]

  return trainX,trainY,testX,testY
# ----------------------------------------------------- #


class StockDataset(Dataset):
  def __init__(self,inputs,outputs):
    self.inputs  = inputs
    self.outputs = outputs

  def __len__(self):
    return len(self.inputs)

  def __getitem__(self,i):
    input  = torch.Tensor(self.inputs[i])
    output = torch.Tensor(self.outputs[i])
    return (input,output)

start_date = datetime.strptime('02/02/18', '%m/%d/%y')
end_date = datetime.strptime('02/09/20', '%m/%d/%y')
dataset = get_dataset(symbol="TATAMOTORS",start_date=start_date,end_date=end_date)
trainX,trainY,testX,testY = prepare_data(dataset,'Close','Date')

train_ds = StockDataset(trainX,trainY)
test_ds  = StockDataset(testX,testY)

# Defining the configs
config = PerceiverConfig(
    input_len = len(train_ds[0][0]),
    input_dim = 1,
    latent_dim = 1,
    num_heads=1,
    output_len = 1,
    output_dim = 1,
    decoder_cross_attention=True,
    decoder_projection=False,
)

# Instantiating model with configs
class StockDataPrediction(torch.nn.Module):
  def __init__(self,config):
    super().__init__()
    self.config = config
    self.emb = build_position_encoding("trainable", config, config.input_len, config.input_dim)
    self.perceiver = Perceiver(config)

  def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

  def forward(self, x):
        pos_emb = torch.cat([self.emb[None, ...] for _ in range(x.shape[0])], dim=0)
        out = x + pos_emb
        return self.perceiver(out)

# Defining the dataloaders
batch_size=32
dl_train = DataLoader(train_ds,batch_size=32,shuffle=True,drop_last=True)
dl_test  = DataLoader(test_ds, batch_size, shuffle=False, drop_last=False)

set_seed(config.seed)
model = StockDataPrediction(config)
print("model parameters:", model.num_parameters())

iter_dl_train = iter(dl_train)

pbar = trange(10000)
optim = Adam(model.parameters(), lr=0.001)
loss_func = torch.nn.MSELoss()
all_loss = []

# train!
for i in pbar:
    try:
        x, y = next(iter_dl_train)
    except StopIteration:
        iter_dl_train = iter(dl_train)
        x, y = next(iter_dl_train)

    optim.zero_grad()
    _y = model(torch.unsqueeze(x,2))
    loss = loss_func(_y, torch.unsqueeze(y,2))
    all_loss.append(loss.item())
    pbar.set_description(f"loss: {np.mean(all_loss[-50:]):.4f}")
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optim.step()

    if (i + 1) % 500 == 0:
        model.eval()
        with torch.no_grad():
            _all_loss = []
            for x, y in dl_test:
                _y = model(torch.unsqueeze(x,2))
                loss = loss_func(_y, torch.unsqueeze(y,2))
                _all_loss.append(loss.item())
            print(f"Test Loss: {sum(_all_loss)}")
        model.train()


# plotting loss & accuracy
plt.figure(figsize=(20, 10))
plt.subplot(1, 2, 1)
plt.plot(savgol_filter(all_loss, window_length=51, polyorder=3))
plt.title("Training Loss")
plt.subplot(1, 2, 2)
plt.plot(_all_loss)
plt.title("Testing Loss")
plt.show()
