"""
Trainer
=======

This is a generic trainer built for the gperc project. More documentation will be added later.
"""

import os
import tarfile
from tqdm.auto import trange
import torch

from .utils import timeit

class Trainer():
  def __init__(
    self,
    model,
    train_data,
    test_data,
    optim,
    save_folder,
    save_every,
    logger_client,
    gpu: int = 0,
    verbose = False,
  ):
    # create model, data and optimizer
    self.model = model
    self.train_data = train_data
    self.test_data = test_data
    self.optim = optim
    
    if verbose:
      print(f"Model params: {self.model.num_parameters()}")
      print(f"Train data: {self.train_data}")
      print(f"Test data: {self.test_data}")
      print(f"Optimizer: {self.optim}")

    self.save_folder = save_folder
    self.save_every = save_every
    self.logger_client = logger_client

    self.model_config = model.config
    self.data_config = train_data.config
    self.device = "cuda:0" if (gpu and torch.cuda.is_available()) else "cpu"
    self.model.to(self.device)

    if self.save_folder != None:
      os.makedirs(self.save_folder, exist_ok = True) # create this just in case
      with open(os.path.join(self.save_folder, "config.json"), "w") as f:
        f.write(self.model_config.to_json())

  def serialise(self, name, optim = None, lr_scheduler = None):
    # create a new folder for current step
    if self.save_folder == None:
      print("No save folder specified, skipping saving.")
      return
    step_folder = os.path.join(self.save_folder, name)
    print(f"Saving in folder: {step_folder}")
    os.makedirs(step_folder, exist_ok = True)
    
    # start saving things -> model, optim, lr_scheduler, (train_data, test_data)
    torch.save(self.model.state_dict(), os.path.join(step_folder, "model.pt"))
    if optim != None:
      torch.save(optim.state_dict(), os.path.join(step_folder, "optim.pt"))
    if lr_scheduler != None:
      torch.save(lr_scheduler.state_dict(), os.path.join(step_folder, "lr_scheduler.pt"))

    # create a tar file of the folder
    fpath = os.path.join(self.save_folder, f"{name}.tar.gz")
    with tarfile.open(fpath, "w:gz") as tar:
      tar.add(step_folder, arcname = name)
    print(f"Saved to {fpath}")
    return fpath

  def load_from_tar(self, fpath):
    with tarfile.open(fpath, "r:gz") as tar:
      tar.extractall(path = self.save_folder)
    folder = os.path.join(self.save_folder, fpath.split("/")[-1].split(".")[0])
    self.model.load_state_dict(torch.load(os.path.join(folder, "model.pt")))
    if os.path.exists(os.path.join(folder, "optim.pt")):
      self.optim.load_state_dict(torch.load(os.path.join(folder, "optim.pt")))
    if os.path.exists(os.path.join(folder, "lr_scheduler.pt")):
      self.lr_scheduler.load_state_dict(torch.load(os.path.join(folder, "lr_scheduler.pt")))

  @classmethod
  def deserialise(cls, fpath):
    pass

  def load(self, save_folder, optim = None, lr_scheduler = None):
    """
    Load the model from the given save folder. If any of these fails, you will have to manually check.

    Args:
      save_folder (str): The folder to load from
      optim (torch.optim.Optimizer): The optimizer to load
      lr_scheduler (torch.optim.lr_scheduler): The lr scheduler to load
    """
    self.model.load_state_dict(torch.load(os.path.join(save_folder, "model.pt")))
    if optim != None and os.path.exists(os.path.join(save_folder, "optim.pt")):
      self.optim.load_state_dict(torch.load(os.path.join(save_folder, "optim.pt")))
    if lr_scheduler != None and os.path.exists(os.path.join(save_folder, "lr_scheduler.pt")):
      self.lr_scheduler.load_state_dict(torch.load(os.path.join(save_folder, "lr_scheduler.pt")))

  def __call__(self, batch, step, n_bytes, n_classes, pbar, train = True, grad_clip = 1.0, optim = None):
    """
    Train or test a batch, returns the current batch meta.

    Args:
      batch (dict): The batch from ``gperc.Consumer`` or ``gperc.ArrowConsumer``
      step (int): The current step number
      n_bytes (int): The number of bytes in the dataset
      n_classes (int): The number of classes in the dataset
      pbar (tqdm.auto.trange): The progress bar
      train (bool): Whether to train or test
      grad_clip (float): The gradient clipping value, defaults to 1.0
      optim (torch.optim.Optimizer): The optimizer to use, must be defined if ``train == True``
    """
    # initial setup
    self.model.train() if train else self.model.eval()
    prefix = "train" if train else "val"
    batch_meta = batch.pop("meta", {}) # create meta dict
    batch_meta[f"{prefix}/step"] = step

    # forward pass
    batch = {k:v.to(self.device) for k,v in batch.items()}
    forward_pass_time, out = timeit(self.model)(batch["input_array"])
    batch_meta[f"{prefix}/forward_pass_time"] = forward_pass_time

    # make is so that any task can be run with this
    out = out.contiguous().view(-1, self.model_config["n_classes"])
    target = batch["class"].contiguous().view(-1)
    loss = torch.nn.functional.cross_entropy(out, target, reduction = "none")
    acc = out.argmax(-1).eq(target).float().mean()
    _mean_loss = loss.mean()

    # capture the meta for the processed batch
    am = batch["attention_mask"]
    bytes_by_class = {c: 0 for c in range(n_classes)}
    loss_by_class, acc_by_class = bytes_by_class.copy(), bytes_by_class.copy()
    for i, c in enumerate(batch["class"].tolist()):
      bytes_by_class[c] += n_bytes * am[i].sum().item()
      loss_by_class[c] += loss[i].item()
      acc_by_class[c] += out[i].argmax(-1).eq(target[i]).float().mean().item()

    # update the logging dict
    bytes_processed = am.sum().item() * n_bytes
    batch_meta.update({
      f"{prefix}/bytes_processed": bytes_processed,
      f"{prefix}/bytes_processed_per_second": bytes_processed / forward_pass_time,
      f"{prefix}/class_wise_bytes_processed": bytes_by_class,
      f"{prefix}/loss_avg" : _mean_loss.item(),
      f"{prefix}/acc_avg" : acc.item(),
      f"{prefix}/loss_class" : loss_by_class,
      f"{prefix}/acc_class" : acc_by_class,
    })

    if train:
      # run the backward pass
      optim.zero_grad()
      _mean_loss.backward()
      
      for p in self.model.parameters():
        if p.requires_grad:
          p.grad.data.clamp_(-grad_clip, grad_clip) # clip gradient values

      backward_pass_time, _ = timeit(optim.step)()
      batch_meta["train/backward_pass_time"] = backward_pass_time
      batch_meta["train/total_time"] = forward_pass_time + backward_pass_time

    pbar.set_description(f"[{prefix}] loss: {_mean_loss.item():.4f} | acc: {acc.item():.4f}")
    return batch_meta

  def train(self, n_steps, test_every):
    """Train model with given optimiser, data and number of steps, optionally provide testing
    material as well.

    Args:
      optim (torch.optim.Optimizer): The optimizer to use
      train_data (``gperc.Consumer/ArrowConsumer``): The training data, batches must be created
      n_steps (int): The number of steps to train for
      test_every (int): The number of steps to train for before testing, defaults to ``n_steps``
      test_data (``gperc.Consumer/ArrowConsumer``): The testing data, batches must be created
    """
    optim = self.optim
    train_data = self.train_data
    test_data = self.test_data

    pbar = trange(n_steps)
    min_loss = float("inf")
    best_model_fpath = None

    for i in pbar:
      batch = train_data.get_next_batch()
      batch_meta = self(
        batch = batch,
        step = i,
        n_bytes = train_data.n_bytes,
        n_classes = train_data.n_classes,
        pbar = pbar,
        grad_clip = 1.0,
        optim = optim,
        train = True
      )

      if i and test_every != None and test_data != None and i % test_every == 0:
        pbar_val = trange(len(test_data._batches))
        
        # we need to capture the meta for the test batches since they can be more than one
        metas = []
        for _ in pbar:
          batch = test_data.get_next_batch()
          batch_meta = self(
            batch = batch,
            step = i,
            n_bytes = test_data.n_bytes,
            n_classes = test_data.n_classes,
            pbar = pbar_val,
            grad_clip = 1.0,
            optim = optim,
            train = False
          )
          metas.append(batch_meta)
        
        # mean over all batches
        test_meta = {}
        for k in metas[0].keys():
          v = [m[k] for m in metas]

          if isinstance(v, list) and isinstance(v[0], dict):
            v2 = {}
            for k2 in v[0].keys():
              v2[k2] = [m[k][k2] for m in metas]
            v = v2
          if isinstance(v, dict):
            test_meta[k] = {k2: [m[k][k2] for m in metas] for k2 in v.keys()}
            test_meta[k] = {k2: sum(v)/len(v) for k2, v in test_meta[k].items()}
          else:
            test_meta[k] = sum(v)/len(v)
        batch_meta.update(test_meta)

        print("val/loss:", test_meta["val/loss_avg"])
        print("val/acc:", test_meta["val/acc_avg"])

        if min_loss > test_meta["val/loss_avg"]:
          min_loss = test_meta["val/loss_avg"]
          best_model_fpath = self.serialise(f"step_{i}", optim, lr_scheduler = None)
        
      # log
      if self.logger_client != None:
        self.logger_client(batch_meta)
    
    return best_model_fpath

