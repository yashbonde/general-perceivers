import os
import sys
import time
import math
import joblib

import torch
from torch import nn
from torch.nn import functional as F
from torch.distributed import rpc
from torch.distributed.pipeline.sync import Pipe

from tempfile import gettempdir, NamedTemporaryFile

from gperc.configs import PerceiverConfig
from gperc import Perceiver, TextConfig
from gperc import Embeddings, EncoderBlock, ProcessorBlock, DecoderBlock

if sys.platform == 'win32':
  print('Windows platform is not supported for pipeline parallelism')
  sys.exit(0)
if torch.cuda.device_count() < 2:
  print('Need at least two GPU devices for this tutorial')
  sys.exit(0)


# ======== model ======== #

# config = PerceiverConfig(
#   input_len=10,
#   input_dim=4,
#   latent_len=3,
#   latent_dim=4,
#   output_len=1,
#   output_dim=4,
#   num_layers=2,
#   n_classes=7,
#   decoder_reduction='mean',
#   decoder_projection=True
# )
# model = Perceiver(config)
# with torch.no_grad():
#   out = model(torch.randn(2, 10, 4))
#   print(out.shape)
#   out = model(torch.randn(2, 10, 4), torch.randn(2, 1, 4))
#   print(out.shape)


# Load and batch data
from torchtext.datasets import AG_NEWS
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

pickle_path = os.path.join(gettempdir(), "gperc_pipe_dist.pkl")
tokenizer = get_tokenizer("basic_english")

if os.path.exists(pickle_path):
  print("--------- Loading from", pickle_path, "---------")
  train_tokens_list, train_labels_list, train_attention_mask_list, test_tokens_list, test_labels_list, test_attention_mask_list, vocab = joblib.load(pickle_path)

  train_tokens = torch.from_numpy(train_tokens_list).long()
  train_labels = torch.from_numpy(train_labels_list).long()
  train_attention_mask = torch.from_numpy(train_attention_mask_list).long()
  test_tokens = torch.from_numpy(test_tokens_list).long()
  test_labels = torch.from_numpy(test_labels_list).long()
  test_attention_mask = torch.from_numpy(test_attention_mask_list).long()

else:

  train_iter = AG_NEWS(gettempdir(), split="train")
  vocab = build_vocab_from_iterator((tokenizer(x[1]) for x in train_iter), specials=["<unk>", "<cls>"])
  vocab.set_default_index(vocab["<unk>"])
  _pad_id = vocab(tokenizer("<unk>"))[0]
  _cls_id = vocab(tokenizer("<cls>"))[0]
  print("vocab size:", len(vocab))

  def data_process(data_iter, maxlen=2048, batch_size = 32):
    tokens = []; labels = []; attention_mask = []; sequence_lengths = []
    for x in data_iter:
      _x = [_cls_id,] + vocab(tokenizer(x[1]))[:maxlen]
      sequence_lengths.append(len(_x))
      attention_mask.append([1] * len(_x))
      _x += [_pad_id] * (maxlen - len(_x))
      attention_mask[-1] += [0] * (maxlen - len(attention_mask[-1]))
      labels.append(x[0])
      tokens.append(_x)

    max_len = max(sequence_lengths)

    tokens = torch.tensor(tokens, dtype=torch.long)[:, :max_len]
    attention_mask = torch.tensor(attention_mask, dtype=torch.long)[:, :max_len]
    labels = torch.tensor(labels, dtype=torch.long)

    n_batches = len(tokens) // batch_size
    tokens = tokens[:n_batches * batch_size].reshape(n_batches, batch_size, -1)
    attention_mask = attention_mask[:n_batches * batch_size].reshape(n_batches, batch_size, -1)
    labels = labels[:n_batches * batch_size].reshape(n_batches, -1)

    return tokens, labels, attention_mask
      
  train_iter, test_iter = AG_NEWS(gettempdir())
  train_tokens, train_labels, train_attention_mask = data_process(train_iter)
  test_tokens, test_labels, test_attention_mask = data_process(test_iter)

  train_tokens_list = train_tokens.numpy()
  train_labels_list = train_labels.numpy()
  train_attention_mask_list = train_attention_mask.numpy()
  test_tokens_list = test_tokens.numpy()
  test_labels_list = test_labels.numpy()
  test_attention_mask_list = test_attention_mask.numpy()

  joblib.dump(
    (train_tokens_list, train_labels_list, train_attention_mask_list, test_tokens_list, test_labels_list, test_attention_mask_list, vocab),
    pickle_path
  )
    

print("=" * 70)
print("        train_tokens:", train_tokens.shape)
print("        train_labels:", train_labels.shape)
print("train_attention_mask:", train_attention_mask.shape)
print("=" * 70)
print("        test_tokens:", test_tokens.shape)
print("        test_labels:", test_labels.shape)
print("test_attention_mask:", test_attention_mask.shape)
print("=" * 70)

train_zip = zip(train_tokens, train_labels, train_attention_mask)
test_zip = zip(test_tokens, test_labels, test_attention_mask)

device = torch.device("cuda")

config = TextConfig(
  latent_dim = 128,
  vocab_size = len(vocab),
  max_len = train_tokens.shape[-1],
  latent_frac = 0.25, # 75% area reduction per layer
  decoder_reduction='first',
  decoder_projection=True,
  n_classes = 4, # hard coded for now, can be modified later
  num_layers = 2
)
print(config)
print("=" * 70)
print("Creating model ...")

tmpfile = NamedTemporaryFile()
rpc.init_rpc(
  name="worker",
  rank=0,
  world_size=1,
  rpc_backend_options=rpc.TensorPipeRpcBackendOptions(
    init_method="file://{}".format(tmpfile.name),
    # Specifying _transports and _channels is a workaround and we no longer
    # will have to specify _transports and _channels for PyTorch
    # versions >= 1.8.1
    _transports=["ibv", "uv"],
    _channels=["cuda_ipc", "cuda_basic"],
  ),
)


num_gpus = torch.cuda.device_count()
partition_len = ((config.num_layers - 1) // num_gpus) + 1

# Add encoder in the beginning.
tmp_list = [Embeddings(config).cuda(0), EncoderBlock(config).cuda(0),]
module_list = []

# Add all the necessary transformer blocks.
for i in range(config.num_layers):
  processor_block = ProcessorBlock(config)
  if i != 0 and i % (partition_len) == 0:
    module_list.append(nn.Sequential(*tmp_list))
    tmp_list = []
  device = i // (partition_len)
  tmp_list.append(processor_block.cuda(device))

# Add decoder in the end.
tmp_list.append(DecoderBlock(config).cuda(num_gpus - 1))
module_list.append(nn.Sequential(*tmp_list))

# Build the pipeline.
chunks = 1
model = Pipe(nn.Sequential(*module_list), chunks=chunks)

def get_total_params(module: torch.nn.Module):
  total_params = 0
  for param in module.parameters():
    total_params += param.numel()
  return total_params

print("Total parameters in model: {:,}".format(get_total_params(model)))

print("=" * 70)
print("Testing forward pass ...")
inputs, target, attn_mask = next(train_zip)
model_input = [
  inputs.cuda(0),
  attn_mask.cuda(0),
  torch.tensor([-420. for _ in range(inputs.shape[0])]).cuda(0)
]
print(model_input)
with torch.no_grad():
  output, attentions = model(*model_input).local_value()

print("output.shape:", output.shape)
for a in attentions[0]:
  print(a.shape)

print("=" * 70)

lr = 5.0  # learning rate
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
loss_fn = nn.CrossEntropyLoss()


def train():
  model.train()  # Turn on the train mode
  total_loss = 0.0
  start_time = time.time()
  ntokens = len(vocab)

  for batch, (inputs, target, attn_mask) in enumerate(train_zip):
    optimizer.zero_grad()
    # Since the Pipe is only within a single host and process the ``RRef``
    # returned by forward method is local to this node and can simply
    # retrieved via ``RRef.local_value()``.
    model_input = (
      inputs.cuda(0),
      attn_mask.cuda(0),
      torch.tensor([-420. for _ in range(inputs.shape[0])]).cuda(0)
    )
    output, attentions = model(*model_input).local_value()
    # Need to move targets to the device where the output of the
    # pipeline resides.
    logits = output.contiguous().reshape(-1, config.vocab_size)
    target = target.contiguous().reshape(-1).cuda(1)
    loss = loss_fn(logits, target)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    optimizer.step()

    total_loss += loss.item()
    log_interval = 10
    if batch % log_interval == 0 and batch > 0:
      cur_loss = total_loss / log_interval
      elapsed = time.time() - start_time
      print(
        "| epoch {:3d} | {:5d}/{:5d} batches | "
        "lr {:02.2f} | ms/batch {:5.2f} | "
        "loss {:5.2f} | ppl {:8.2f}".format(
          epoch, batch, batch // len(train_tokens),
          scheduler.get_lr()[0], elapsed * 1000 / log_interval,
          cur_loss, math.exp(cur_loss)
        )
      )
      total_loss = 0
      start_time = time.time()


def evaluate(eval_model):
  eval_model.eval()  # Turn on the evaluation mode
  total_loss = 0.0
  # Evaluate only for 50 batches to keep script execution time low.
  with torch.no_grad():
    for (inputs, target, attn_mask) in test_zip:
      model_input = (
        inputs.cuda(0),
        attn_mask.cuda(0),
        torch.tensor([-420. for _ in range(inputs.shape[0])]).cuda(0)
      )
      output, attentions = eval_model(*model_input).local_value()
      logits = output.contiguous().reshape(-1, config.vocab_size)
      target = target.contiguous().reshape(-1).cuda(1)
      loss = loss_fn(logits, target)
      # Need to move targets to the device where the output of the
      # pipeline resides.
      total_loss += len(logits) * loss.item()
  return total_loss / (len(test_zip) - 1)


best_val_loss = float("inf")
epochs = 3  # The number of epochs
best_model = None


for epoch in range(1, epochs + 1):
  epoch_start_time = time.time()
  train()
  val_loss = evaluate(model)
  print("-" * 89)
  print(
    "| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | "
    "valid ppl {:8.2f}".format(epoch, (time.time() - epoch_start_time), val_loss, math.exp(val_loss))
  )
  print("-" * 89)

  if val_loss < best_val_loss:
    best_val_loss = val_loss
    best_model = model

  scheduler.step()


test_loss = evaluate(best_model)
print("=" * 89)
print("| End of training | test loss {:5.2f} | test ppl {:8.2f}".format(test_loss, math.exp(test_loss)))
print("=" * 89)
