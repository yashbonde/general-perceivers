# example with DataDistributed
from tqdm.auto import tqdm
from tempfile import gettempdir

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

from torchvision import transforms as TR
from torchvision.datasets import CIFAR10

# -----
from gperc.utils import set_seed
from gperc import ImageConfig, PerceiverImage
# -----

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
    latent_len=128,
    latent_dim=1024,
    n_classes=10,
    num_layers = 16,
    num_heads = 16
)

def train_fn():
    
    # create model and move it to GPU with id rank
    model = PerceiverImage(config)
    print(f"Number of parameters in the model: {model.num_parameters():,}")
    device = torch.cuda.current_device()
    ddp_model = torch.nn.DataParallel(model).to(device)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = Adam(ddp_model.parameters(), lr=0.001)

    # define the dataloaders, optimizers and lists
    batch_size = 256
    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True, drop_last=True)
    dl_test = DataLoader(ds_test, batch_size=batch_size, shuffle=False, drop_last=False)

    n_epochs = 10
    for epoch in range(n_epochs):
        # first train
        pbar = tqdm(enumerate(dl_train), total=len(dl_train))
        for i, (x, y) in pbar:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            x_hat = ddp_model(x)
            loss = loss_fn(x_hat, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), 1.0)
            optimizer.step()

            acc = (x_hat.argmax(dim=1) == y).float().mean()
            pbar.set_description(f"Epoch: {epoch}, Loss: {loss.item():.4f}, Acc: {acc.item():.4f}")

        # then test
        with torch.no_grad():
            ddp_model.eval()
            all_loss = []; all_acc = []
            pbar = tqdm(enumerate(dl_test), total=len(dl_test))
            for i, (x, y) in pbar:
                x = x.to(device)
                y = y.to(device)
                x_hat = ddp_model(x)
                loss = loss_fn(x_hat, y)
                acc = (x_hat.argmax(dim=1) == y).float().mean()
                all_loss.append(loss.item())
                all_acc.append(acc.item())
                pbar.set_description(f"Epoch: {epoch}, Loss: {loss.item():.4f}, Acc: {acc.item():.4f}")
            print(f"Epoch: {epoch}, Test loss: {sum(all_loss):.4f}, test acc: {sum(all_acc)/len(all_acc):.4f}")
            ddp_model.train()

train_fn()
