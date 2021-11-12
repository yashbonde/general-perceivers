# Data preparation functions from https://github.com/PacktPublishing/PyTorch-Computer-Vision-Cookbook/blob/master/Chapter10/myutils.py

import os
import pathlib
import subprocess

import glob
from PIL import Image
from tqdm import trange
import cv2
import numpy as np
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit

import torch
import torchvision.transforms as transforms
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam

from gperc.utils import set_seed
from gperc import PerceiverConfig, Perceiver
from gperc.models import build_position_encoding

# labels in ucf11
labels = ["basketball","biking","diving","golf_swing","horse_riding","soccer_juggling","swing","tennis_swing","trampoline_jumping","volleyball_spiking","walking"]
labels_dict = {}

# ------------- Utility Functions ---------------- #
def get_frames(filename, n_frames= 1):
    frames = []
    v_cap = cv2.VideoCapture(filename)
    v_len = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_list= np.linspace(0, v_len-1, n_frames, dtype=np.int16)
    for fn in range(v_len):
        success, frame = v_cap.read()
        if success is False:
            continue
        if (fn in frame_list):
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
    v_cap.release()
    return frames, v_len

def store_frames(frames, path2store):
    for ii, frame in enumerate(frames):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        path2img = os.path.join(path2store, "frame"+str(ii)+".jpg")
        cv2.imwrite(path2img, frame)

def get_vids(path2ajpgs):
    listOfCats = os.listdir(path2ajpgs)
    ids = []
    labels = []
    for catg in listOfCats:
        path2catg = os.path.join(path2ajpgs, catg)
        listOfSubCats = os.listdir(path2catg)
        path2subCats= [os.path.join(path2catg,los) for los in listOfSubCats]
        ids.extend(path2subCats)
        labels.extend([catg]*len(listOfSubCats))
    return ids, labels, listOfCats

def get_labels(catgs):
    global labels_dict
    ind = 0
    for uc in catgs:
        labels_dict[uc] = ind
        ind+=1
    return labels_dict
# ------------------------------------------------ #

# Preparing the dataset
def prepare_data(n_frames=10, num_classes=5):
    url = "https://www.crcv.ucf.edu/data/YouTube_DataSet_Annotated.zip"
    subprocess.run(["wget", "--no-check-certificate",url])
    subprocess.run(["unzip","YouTube_DataSet_Annotated.zip","-d","."])

    os.mkdir('dataset')
    for label in labels:
        os.mkdir("dataset/"+label)
    for root, dirs, files in os.walk("./action_youtube_naudio"):
        for file in files:
            if file[-4:]=='.avi':
                dir=os.path.dirname(root)
                pathlib.Path(root+"/"+file).rename("dataset/"+dir.split("/")[-1]+"/"+file)

    for root, dirs, files in os.walk("./dataset", topdown=False):
        for name in files:
            if ".avi" not in name:
                continue
            path2vid = os.path.join(root, name)
            frames, vlen = get_frames(path2vid, n_frames= n_frames)
            if len(frames) != n_frames:
                continue
            path2store = path2vid.replace("dataset", "image_dataset")
            path2store = path2store.replace(".avi", "")
            os.makedirs(path2store, exist_ok= True)
            store_frames(frames, path2store)

    all_vids, all_labels, catgs = get_vids('image_dataset')
    labels_dict = get_labels(catgs)
    unique_ids = [id_ for id_, label in zip(all_vids,all_labels) if labels_dict[label]<num_classes]
    unique_labels = [label for id_, label in zip(all_vids,all_labels) if labels_dict[label]<num_classes]

    sss = StratifiedShuffleSplit(n_splits=2, test_size=0.1, random_state=0)
    train_indx, test_indx = next(sss.split(unique_ids, unique_labels))
    train_ids = [unique_ids[ind] for ind in train_indx]
    train_labels = [unique_labels[ind] for ind in train_indx]
    test_ids = [unique_ids[ind] for ind in test_indx]
    test_labels = [unique_labels[ind] for ind in test_indx]

    return train_ids, train_labels, test_ids, test_labels


class UC11Dataset(Dataset):
    def __init__(self, ids, labels, transform):
        self.transform = transform
        self.ids = ids
        self.labels = labels

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        path2imgs=glob.glob(self.ids[idx]+"/*.jpg")
        path2imgs = path2imgs[:10]
        label = labels_dict[self.labels[idx]]
        frames = []
        for p2i in path2imgs:
            frame = Image.open(p2i)
            frames.append(frame)
        seed = np.random.randint(1e9)
        frames_tr = []
        for frame in frames:
            frame = self.transform(frame)
            frames_tr.append(frame)
        if len(frames_tr)>0:
            frames_tr = torch.stack(frames_tr)
        return frames_tr, label


train_transformer = transforms.Compose([
            transforms.Resize((32,32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            transforms.Lambda(lambda x: x.permute(1, 2, 0).reshape(-1, 3)),
            ])

test_transformer = transforms.Compose([
            transforms.Resize((32,32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            transforms.Lambda(lambda x: x.permute(1, 2, 0).reshape(-1, 3)),
            ])


# Defining the spatial and temporal configs
spatialConfig = PerceiverConfig(
    input_len =32*32,
    input_dim = 3,
    latent_len = 32,
    latent_dim = 32,
    output_len = 1,
    output_dim = 64,
    decoder_cross_attention=True,
    decoder_projection=False,
)

temporalConfig = PerceiverConfig(
    input_len = 10,
    input_dim=64,
    latent_len = 32,
    latent_dim = 32,
    output_len = 32,
    output_dim = 32,
    n_classes = 5,
    decoder_residual = False,
    decoder_projection=True
)

# Defining the model
class UCF11ActionClassifier(torch.nn.Module):
  def __init__(self,spatialConfig,temporalConfig):
    super().__init__()
    self.spatialConfig = spatialConfig
    self.temporalConfig = temporalConfig
    self.spatialEmb = build_position_encoding("trainable", spatialConfig, spatialConfig.input_len, spatialConfig.input_dim)
    self.temporalEmb = build_position_encoding("trainable", temporalConfig, temporalConfig.input_len, temporalConfig.input_dim)
    self.spatialEncoder = Perceiver(spatialConfig)
    self.temporalDecoder = Perceiver(temporalConfig)

  def num_parameters(self):
    return sum(p.numel() for p in self.parameters() if p.requires_grad)

  def forward(self,x):
    spatialSequence = []
    for i in range(x.size(1)):
      spatial_pos_emb = torch.cat([self.spatialEmb[None, ...] for _ in range(x.shape[0])], dim=0)
      spatialOut = x[:,i,:,:]+spatial_pos_emb
      spatialSequence.append(self.spatialEncoder(spatialOut))
    spatialSequence = torch.stack(spatialSequence, dim=0)
    spatialSequence = torch.reshape(spatialSequence,(x.shape[0],10,64))
    temporal_pos_emb = torch.cat([self.temporalEmb[None, ...] for _ in range(spatialSequence.shape[0])], dim=0)
    temporalOut = spatialSequence + temporal_pos_emb
    return self.temporalDecoder(temporalOut)


# Instantiating model with configs
set_seed(spatialConfig.seed)
model = UCF11ActionClassifier(spatialConfig,temporalConfig)
print("model parameters:", model.num_parameters())

# Defining the dataloaders
train_ids, train_labels, test_ids, test_labels = prepare_data()
train_ds = UC11Dataset(ids= train_ids, labels= train_labels, transform= train_transformer)
test_ds = UC11Dataset(ids= test_ids, labels= test_labels, transform= test_transformer)
train_dl = DataLoader(train_ds, batch_size= 32,
                          shuffle=True)
test_dl = DataLoader(test_ds, batch_size= 32,
                         shuffle=False)


iter_dl_train = iter(train_dl)
pbar = trange(10000)
optim = Adam(model.parameters(), lr=0.001)
all_loss = []
all_acc = []

# training the model!
for i in pbar:
    try:
        x, y = next(iter_dl_train)
    except StopIteration:
        iter_dl_train = iter(train_dl)
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
            for x, y in test_dl:
                _y = model(x)
                loss = F.cross_entropy(_y, y)
                _all_loss.append(loss.item())
                _all_acc.append((_y.argmax(-1) == y).sum().item() / len(y))
            print(f"Test Loss: {sum(_all_loss)} | Test Acc: {sum(_all_acc)/len(_all_acc)}")
        model.train()


# plotting loss & accuracy
plt.figure(figsize=(20, 10))
plt.subplot(1, 2, 1)
plt.plot(savgol_filter(all_acc, window_length=51, polyorder=3))
plt.title("Training Accuracy")
plt.subplot(1, 2, 2)
plt.plot(all_loss)
plt.title("Training Loss")
plt.show()
