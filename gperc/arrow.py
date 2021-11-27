"""
PyArrow Consumer
================

Using ``gperc.Consumer`` has serious downsides because it is very slow because it has to read
huge number of files and making OS calls again and again is ridiculously expensive. This mandated
the use of something that can store large amounts of information on disk and read write a huge
speeds. I explored using `leveldb <https://github.com/google/leveldb>`_ as key value store, but
it would require writing code for things like applying ``map`` operations, etc. and  there already
is huggigface datasets which can already do that.

This is much faster than using ``gperc.Consumer`` here are some numbers on M1-Air:

.. code-block:: bash

  [100000] 0.011600 ms:   open(fp, 'rb').read()
  [100000] 0.036084 ms:   [x for x in open(fp, 'rb').read()]
  [100000] 0.021213 ms:   list(open(fp, 'rb').read())

  # Random read of 10000 samples is ~5.4x faster
  [010000] 2.621051 ms:   data[random.choice(range(len(data)))]
  [010000] 0.482127 ms:   cons_data[random.choice(range(len(data)))]

  # Sampling at batch size 128 samples for 1000 iterations = 128000 samples
  # which is ~2.13 epochs is ~7.4x faster
  [001000] 430.659158 ms: data.get_next_batch('supervised')
  [001000]  58.214403 ms: cons_data.get_next_batch()


Documentation
-------------
"""

import os
import json
import torch
import random
import hashlib
import datasets
import numpy as np
import pyarrow as pa
from dataclasses import dataclass

from .data import get_vocab, convert_to_gperc_consumer_ir
from .utils import set_seed

logger = datasets.utils.logging.get_logger(__name__)


@dataclass
class BinaryBuilderConfig(datasets.BuilderConfig):
  """Config file for our Arrow Builder"""
  def create_config_id(self, *a, **b):
    """returns the id (hash) for this builder. We use a simple SHA256 dump of the JSON
    string of ``data_files``."""
    _hash = hashlib.sha256(json.dumps(self.data_files).encode()).hexdigest()
    return _hash


class BinaryArrowBuilder(datasets.ArrowBasedBuilder):
  """ArrowBuilder for our Binary dataset object"""
  BUILDER_CONFIG_CLASS = BinaryBuilderConfig

  def _info(self):
    return datasets.DatasetInfo(features=None)

  def _split_generators(self, *a, **b):
    """This function takes the ``self.datafiles`` and returns the split. ``datafiles`` is in F3 format

      #. **(F3)** dict of categories (IR): ``{"cat1": ["file1.txt", "file2.txt", ...], "cat2": ["file3.txt", "file4.txt", ...]}``
    """
    data_files = self.config.data_files
    files = []; labels = [] # flatten data_files as file <> label pairs
    for c, _f in data_files.items():
      files.extend(_f)
      labels.extend([c for _ in range(len(_f))])

    splits = [
      datasets.SplitGenerator(name = "train", gen_kwargs={
        "files": files,
        "labels": labels
      })
    ]
    return splits

  def _generate_tables(self, files, labels):
    """This function takes in the files and labels and yields a ``PyArrow.Table`` object"""
    schema = pa.schema({
      "binary": pa.binary(),
      "filepath": pa.string(),
      "labels": pa.int64(),
    })
    for file_idx, (file, _label) in enumerate(zip(files, labels)):
      batch_idx = 0
      batch = []
      with open(file, "rb") as f:
        batch.append(file.encode("utf-8") + b"|" + f.read())

      pa_table = pa.Table.from_arrays([pa.array(batch), pa.array([file]), pa.array([_label])], schema=schema)
      # Uncomment for debugging (will print the Arrow table size and elements)
      # logger.warning(f"pa_table: {pa_table} num rows: {pa_table.num_rows}")
      # logger.warning('\n'.join(str(pa_table.slice(i, 1).to_pydict()) for i in range(pa_table.num_rows)))
      yield (file_idx, batch_idx), pa_table
      batch_idx += 1


class ArrowConsumer():
  def __init__(
    self,
    fps,
    n_bytes=2,
    seqlen="auto",
    class_to_id = None,
  ):
    """Read `Consumer <gperc.data.html>`_ for documentation. This class uses PyArrow as underlying table
    however tokenisation etc. still has to be done for each sample when using ``__getitem__``. This method
    exclusively works for classification problems (``style=='diff'``)."""
    self.fps, self._mode = convert_to_gperc_consumer_ir(fps)
    self.class_to_id = class_to_id
    self.id_to_class = None

    if class_to_id != None:
      try:
        # try to convert to int, if it works then it is already correct
        int(next(self.fps))
      except:
        # check that the keys match
        keys_from_id = set(class_to_id.keys())
        keys_from_fps = set(self.fps.keys())
        assert keys_from_id - keys_from_fps == set(), f"Mismatch keys: {keys_from_id - keys_from_fps}"
        self.fps = {class_to_id[k]: v for k, v in self.fps.items()}
      self.id_to_class = {v: k for k, v in self.class_to_id.items()}

    # values set for ops
    vocab_size = int(2 ** (8 * n_bytes))
    self.vocab = get_vocab(n_bytes)
    self.__auto_idx = 0
    self.n_classes = len(self.fps)
    self.style = "diff"
    self.n_bytes = n_bytes
    self.seqlen = seqlen
    self.vocab_size = vocab_size + 1

    # vocabulary building process special tokens
    self.EOF_ID = self.vocab_size - 1
    self.EOF_TOKEN = "<EOF>"
    self.vocab[self.EOF_TOKEN] = self.EOF_ID

    builder_instance = BinaryArrowBuilder(name="my_datasets", data_files=self.fps, hash=hash,)
    builder_instance.download_and_prepare(try_from_hf_gcs = False)
    data = builder_instance.as_dataset(in_memory=datasets.info_utils.is_small_dataset(builder_instance.info.dataset_size))

    # map the data to tokenize the text

    def __tokenize(examples):
      input_array = []
      for b in examples["binary"]:
        tokens = list(b)
        tokens += [0,] * (self.n_bytes - (len(tokens) % self.n_bytes))
        zip_items = [tokens[i :: self.n_bytes] for i in range(self.n_bytes)]
        tokens = list(zip(*zip_items))
        tokens = [self.vocab[x] for x in tokens]
        if isinstance(self.seqlen, int):
          tokens = tokens[:self.seqlen-1] # -1 for EOF
        input_array.append(tokens)
      return {"input_array": input_array}

    num_proc = min(os.cpu_count(), len(data["train"]))
    print(f"Tokenising the entire datset, this can takes some time (will use {num_proc} cores) ...")
    data = data.map(
      function = __tokenize,
      num_proc = num_proc,
      batched = True,
      batch_size = 2 << 8
    )
    self.seqlen = max([len(x) for x in data["train"]["input_array"]]) +1 # +1 to compensate for EOF
    self.all_samples = data["train"]
    self.sample_by_class = {k: list(range(len(v))) for k, v in self.fps.items()}

  def __len__(self):
    return len(self.all_samples)

  def get_dict(self):
    _d = {
      "total_samples": len(self),
      "mode": self._mode,
      "n_classes": self.n_classes,
      "n_bytes": self.n_bytes,
      "seqlen": self.seqlen,
      "vocab_size": self.vocab_size,
      "style": self.style,
    }
    if hasattr(self, "batch_size"):
      _d["batch_size"] = self.batch_size
    try:
      _d["total_batches"] = len(self._batches)
    except:
      pass
    return _d

  def to_json(self, fp = None):
    _j = json.dumps(self.get_dict(), indent=2)
    if fp != None:
      with open(fp, "w") as f:
        f.write(_j)
    return _j

  def __repr__(self):
    return f"<gperc ArrowConsumer {self.to_json()}>"

  def __getitem__(self, x):
    """Behaviour should be same as ``gperc.Consumer``
    
    Using this is very simple.

        .. code-block:: python

            # define the consumer object
            my_kewl_dataset = Consumer(
                fps = {
                    "cat": ["img0.png", "/tmp/ssg3hng.png", ...],
                    "dog": ["img1.png", "/tmp/uo35523.png", ...],
                },
                seed = 4
            )

            # output in all cases is a batched tensor of desired shape
            out = my_kewl_dataset[None] # get whatever is the next batch
            out = my_kewl_dataset[0]    # get the data at index 0
            out = my_kewl_dataset[5:10] # get the data at indices 5 to 10
            out = my_kewl_dataset[{
                "cat": 10,
                "dog": 4
            }] # return random batches of 10 samples from class cat and 4 samples from class dog
            out = my_kewl_dataset[{
                "cat": [0, 1, 2, 3, 4],
                "dog": [5, 6, 7, 8, 9]
            }] # return the batches at indices [0...4] and [5...9] from class cat and class dog respectively

            # in all cases above out is a dict with key "input_array" because we have not provided a query
            # if you want to force this behaviour
            out = my_kewl_dataset[5:10, None]
    """
    all_samples = self.all_samples
    sample_by_class = self.sample_by_class

    if x == None:
      batch_data = all_samples[self.__auto_idx]
      self.__auto_idx += 1
      if self.__auto_idx == len(self):
        self.__auto_idx = 0

    elif isinstance(x, int):  # i1
      batch_data = all_samples[x]

    elif isinstance(x, (slice, list, tuple)):  # i2, i3
      batch_data = all_samples[x]

    elif isinstance(x, dict):
      if len(self.fps) == 1 and "null" in self.fps:
        raise ValueError("There is no category provided, so you cannot try to make a batch from a dict")
      assert isinstance(list(x.values())[0], (int, list)), f"Values in dict must be integers or lists"
      x = {self.class_to_id[k]: v for k, v in x.items()}

      keys_in_x_not_in_fps = set(x.keys()).difference(set(self.fps.keys()))
      assert keys_in_x_not_in_fps == set(), f"Keys in dict must be in fps: {keys_in_x_not_in_fps}"
      batch_idx = []
      for k, v in x.items():
        sample_idx = sample_by_class[k]
        if isinstance(v, int):  # i4
          assert v > 0, f"Values in dict must be positive integers"
          batch_idx.extend(np.random.choice(sample_idx, v, replace=False).tolist())
        elif isinstance(v, list):  # i5
          batch_idx.extend([sample_idx[i] for i in v])

      batch_data = all_samples[batch_idx]

    # get the data and tokenize it and all
    tokens = batch_data["input_array"]
    attention_mask = []
    input_array = []
    if isinstance(tokens[0], (tuple, list, slice)):
      for j in range(len(tokens)):
        attention_mask.append([1] * (len(tokens[j]) + 1) + [0] * (self.seqlen - len(tokens[j]) - 1))
        input_array.append(tokens[j] + [self.EOF_ID] * (self.seqlen - len(tokens[j])))
      labels = batch_data["labels"]
    else:
      attention_mask = [[1] * (len(tokens) + 1) + [0] * (self.seqlen - len(tokens) - 1)]
      input_array = [tokens + [self.EOF_ID] * (self.seqlen - len(tokens))]
      labels = [batch_data["labels"]]

    # convert to tensors and return the dict
    input_array = torch.tensor(input_array, dtype = torch.long)
    attention_mask = torch.tensor(attention_mask, dtype = torch.long)
    labels = torch.tensor(labels, dtype = torch.long)
    return {
      "input_array": input_array,
      "attention_mask": attention_mask,
      "class": labels
    }

  def create_batches(self, batch_size, drop_last=False, seed=4):
    set_seed(seed)
    samples = list(range(len(self.all_samples)))
    random.shuffle(samples)
    batches = [samples[i : i + batch_size] for i in range(0, len(samples), batch_size)]
    if len(batches[-1]) != batch_size and drop_last:
        batches = batches[:-1]

    # define so can be used later
    self.batch_size = batch_size
    self.drop_last = drop_last
    self.seed = seed

    # define the data points
    self._batches = batches
    self._iter_batches = iter(batches)
    self.__batch_mode = True

  # def batch_analysis(self, batch):
  #   am = batch["attention_mask"] # studying this gives all the information we need
  #   total_bytes = am.sum().item() * self.n_bytes

  #   bytes_by_class = {c: 0 for c in range(self.n_classes)}
  #   for i, c in enumerate(batch["class"].tolist()):
  #       bytes_by_class[c] += self.n_bytes * am[i].sum().item()

  #   return total_bytes, bytes_by_class

  def get_next_batch(self):
    """If you set this mode, you get extra information in key meta against each batch"""
    assert self.__batch_mode, "You must create batches first data.create_batches()"
    try:
      x = next(self._iter_batches)
    except StopIteration:
      # shuffle the batches at the end with a new seed
      self.create_batches(self.batch_size, self.drop_last, self.seed + 1)
      x = next(self._iter_batches)

    data = self[x]
    return data
