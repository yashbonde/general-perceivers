"""
PyArrow Consumer
================

Using ``gperc.Consumer`` has serious downsides because it is very slow because it has to read
huge number of files and making OS calls again and again is ridiculously expensive. This mandated
the use of something that can store large amounts of information on disk and read write a huge
speeds. I explored using `leveldb <https://github.com/google/leveldb>`_ as key value store, but
it would require writing code for things like applying ``map`` operations, etc. and  there already
is huggigface datasets which can already do that.


The general idea is that the trainer should be able to select the kind of data that it wants to use.
This means that there needs to be a structured way to represent and fetch the information.
This is done as follows:

#. The input data ``F`` can be loaded in 4 different styles as given in the documentation below.
#. The fetching ``I`` can happen in 6 different styles as given in the documentation below.

I am following the same system that I have implemented in
`nbox.Parsers <https://nimbleboxai.github.io/nbox/nbox.parsers.html>`_. Here is a quick brief on
**primitives** ``P`` and **structures** ``S``:

#. ``P`` are the basic data types that are used in the data. This is the actual data \
    you want your model to process.
#. ``S`` are the structures that are used to represent the data. This is how data is organised.

In our day to day life, what we call data is nothing but an interplay of one ``P`` and many ``S``.

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

import io
import os
import json
from types import SimpleNamespace
import torch
import random
import hashlib
import datasets
import subprocess
import numpy as np
import pyarrow as pa
from glob import glob
from itertools import product
from functools import lru_cache
from dataclasses import dataclass

from typing import Callable, List


from .utils import set_seed

logger = datasets.utils.logging.get_logger(__name__)


@lru_cache()
def get_vocab(n_bytes):
  B = range(2 ** 8)
  out = product(B, repeat=n_bytes)
  vocab = {x: i for i, x in enumerate(list(out))}
  return vocab


def convert_to_gperc_consumer_ir(fps):
  mode = None
  if isinstance(fps, list):
    if isinstance(fps[0], str):  # F0
      fps = {"null": fps}  # list of files will start with null category
      mode = "F0"
    elif isinstance(fps[0], dict):  # F1
      _fps = {}
      for x in fps:
        k = list(x.keys())[0]
        v = list(x.values())[0]
        _fps.setdefault(v, []).append(k)  # list of dicts will start with category as key
      fps = _fps
      mode = "F1"
    else:
      raise ValueError("fps is not in the correct format")
  elif isinstance(fps, dict):
    k = next(iter(fps))
    v = fps[k]
    assert isinstance(k, str), f"key has to be a string got: {type(k)}"
    if isinstance(v, str):  # F2
      assert all([isinstance(_v, str) for _k, _v in fps.items()]), "All values should be a string"
      _fps = {}
      for k, v in fps.items():
        _fps.setdefault(v, []).append(k)
      fps = _fps
      mode = "F2"
    elif isinstance(v, list):  # F3
      # this is the format we want so just perform checks
      assert all([isinstance(_v, list) for _k, _v in fps.items()]), "All values should be a list"
      mode = "F3"
  else:
      raise ValueError(f"fps is not in the correct format: {type(fps)}")
  
  # sort fps on values then keys
  fps = {k: sorted(v) for k, v in fps.items()}
  fps = dict(sorted(fps.items(), key=lambda x: x[0]))
  return fps, mode


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


class ArrowConfig:
  def __init__(self, fps = None, data_path = None, n_bytes = 2, seqlen = "auto", class_to_id = None, callbacks: List[Callable] = []):
    """Config for Arrow Dataset Reader.

    Args:
        fps ([type], optional): [description]. Defaults to None.
        data_path ([type], optional): [description]. Defaults to None.
        n_bytes (int, optional): [description]. Defaults to 2.
        seqlen (str, optional): [description]. Defaults to "auto".
        class_to_id (Dict[str, int], optional): [description]. Defaults to None.
        callbacks (List[Callable], optional): should be a function that recieves the ``ArrowConsumer`` object
          and performs inplace transformations
    """
    self.fps = fps
    self.data_path = data_path
    self.n_bytes = n_bytes
    self.seqlen = seqlen
    self.class_to_id = class_to_id
    self.callbacks = callbacks

    self.splits = {
      "_def": {
        "fps": self.fps,
        "data_path": self.data_path,
        "n_bytes": self.n_bytes,
        "seqlen": self.seqlen,
        "class_to_id": self.class_to_id,
        "callbacks": self.callbacks
      }
    }
  
  def __init(self, split, **kwargs):
    self.splits.setdefault(split, {})
    self.splits[split].update(kwargs)
    return self

  def split(self, cls):
    full_splits = {}
    for sp in self.splits:
      full_splits[sp] = self.splits["_def"].copy()
      full_splits[sp].update(self.splits[sp])
      full_splits[sp] = SimpleNamespace(**full_splits[sp])
    del full_splits["_def"]
    return SimpleNamespace(**{
      x: cls(full_splits[x]) for x in full_splits
    })

  def __getattr__(self, __name):
    if __name.startswith("_") and __name.endswith("_"):
      from functools import partial
      return partial(self.__init, __name[1:-1])
    else:
      raise AttributeError(f"{__name} is not a valid attribute")

class ArrowConsumer():
  def __init__(
    self,
    config: ArrowConfig,
  ):
    """This class uses PyArrow as underlying table. Since ``ArrowConsumer`` by default uses huggingface ``datasets``
    that manages caching, we do not need to add any heavy overhead of serialisation for the sake of simplicity.

    Consumer takes in list of files along with it's meta data and becomes a callable generator.
    When calling you can tell it what kind of data that you want. It is a full fledged data engine in itself.
    This will sit in nbox one day and thus has to be engineered in such a what that it is production grade with
    good documentation. In the nbox hierarchy it sits parallel to nbox.Model thus has to continue the following
    traits as `nbox.Parsers <https://nimbleboxai.github.io/nbox/nbox.parsers.html>`_:

    #. **primitive** that tells the actual fetching instruction
    #. **structure** should be same as the source meta data

    This ``Consumer`` object will convert any input to the F3 format as internal representation. Moreover for
    each file we will extract the token sequence, the target token sequence looks like this:

    .. code-block:: python

        sequence = [tokens,from,meta,data] + [tokens,from,actual,file] + [EOF-tag]

    This will be the input to the model and this is the final version, this provides sufficient context to the
    model for the given input just like how much information OS has about any given file. The meta data is
    obtained using ``file`` command on posix systems (`man page <https://linux.die.net/man/1/file>`_).

    The file paths have to be the primary index inside the lists and so filepaths "fps" can look like these:

    #. **(F0)** list of strings: ``["file1.txt", "file2.txt", ...]``
    #. **(F1)** list of dicts: ``[{"file1.txt": "cat1"}, {"file2.txt": "cat2"}, ...]``
    #. **(F2)** dict of strings: ``{"file1.txt": "cat1", "file2.txt": "cat2", ...}``
    #. **(F3)** dict of categories (IR): ``{"cat1": ["file1.txt", "file2.txt", ...], "cat2": ["file3.txt", "file4.txt", ...]}``


    Note that in above case ``st_size`` will also include the metadata tokens.        

    .. notes

        `Idempotency <https://en.wikipedia.org/wiki/Idempotence>`_ is a very important so every operation must be the
        same for a long time to come. Given this I am trying to make it as useful and generic as possible. When the data
        is read we add 1 bytes more than ``os.stat`` for ``"<EOF>"`` for end of the sequence.
    """
    self.config = config

    data_path = config.data_path
    fps = config.fps
    if (data_path and fps) or (not data_path and not fps):
      raise ValueError("data_path or fps must be specified")

    if data_path and os.path.isdir(data_path):
      proc = subprocess.Popen(["ls", "-la", data_path], stdout=subprocess.PIPE)
      lines = io.TextIOWrapper(proc.stdout, encoding="utf-8")
      all_lines = "".join(lines).split("\n")[1:-1]
      folders = [l for l in all_lines if l.startswith("d") if not l.endswith(".")]
      files = [l for l in all_lines if l.startswith("-")]
      if folders:
        assert not len(files), "Either folders or files should be present, not both"
        fps = {f: glob(os.path.join(data_path, f, "*")) for f in folders}
      elif files:
        assert not len(folders), "Either folders or files should be present, not both"
        fps = files
      else:
        raise ValueError("No files or folders found in data_path")
    elif data_path and os.path.isfile(data_path):
      fps = [data_path]
    elif fps:
      pass
    else:
      raise ValueError(f"Could not load data from {data_path}")

    # check the config and load items
    n_bytes = config.n_bytes
    seqlen = config.seqlen
    class_to_id = config.class_to_id

    # original loading method
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

    builder_instance = BinaryArrowBuilder(name="my_datasets", data_files=self.fps)
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

    # run callbacks
    for op in config.callbacks:
      op(self)


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
    r"""This is the heart of this code, it takes in user requests and returns the data according to it. This is slightly
    technical and so we will explain it in detail. I find similarities between databases in CRUD and datasets for machine
    learning, CRUD has amazing performance and interaction tools like SQL. Datasets in ML are more like a collection of
    data, and they are not designed to be used in friendly way. Everyone's writing their own thing there but good UX requires
    satisfying the user in some kind of formula and then let them be.

    Any SQL query has the following grammar ``SELECT [columns] FROM [table] WHERE [condition]``. This is something everyone
    understands, it's very simple. In our case ``[table] == self``, i.e. the table is the dataset itself, this is no RDMS.
    The condition is very clearly described in the documentation of ``x``. But ``[columns]`` (here calling it ``query``) is
    something hard, ie. user needs something in a particular format, and with random user logic is hard to give guarantees.
    I will come back to this later.

    The ``condition``, has two parts, the ``primitive`` and ``structure``. With this version of the code, the ``structure``
    and ``primitive`` are implemented in pythonic way. Read the documentation of ``x`` for more details. After getting the data
    we convert it to an intermediate format, which is a list of tuples, each tuple is a sample. The intermediate format has the
    can be one of the following:

    1. dict like this:

    .. code-block:: python

        {
            'data': [
                ('some/file/1', seek_location, end_bytes),
                # >= 1 sample of the above tuple
        ],
            'class': 'tinker'
        }

    2.  list with dict in it, in which case the samples are batched together.

    The intermediate format is then converted to the desired format i.e. ``query``, currently I have added functionality that
    can return one of the following formats:

    1. ``supervised``, in which input is the input tensor and output is the class tensor, from ``self.class_to_id`` dict.
    2. ``unsupervised``, in which input is the input tensor and output is clone of it.


    Args:

        x(Any): There is only one input since this is a special method. We take in this input item and process it accordingly
            based on following rules:

            1. **(I0)** ``None``: when x is None we have an internal idx that is incremented and the next batch is returned
            2. **(I1)** ``int``: when x is an int we return the batch at that index
            3. **(I2)** ``slice``: when x is a slice we return the batches in the slice
            4. **(I3)** ``list``: when x is a list we return the batches in the list containing the indices (``int``)
            5. **(I4)** ``dict -> ints``: when values of x are ints we return the batches in the list containing the indices (``int``)
            6. **(I5)** ``dict -> list``: when values of x are lists we return the batches in the list containing the indices (``list``)
            7. **(I6)** ``tuple``: Read below.

        x_tuple(Tuple): When x is a tuple you can use it like a function, meaning it can run certain hardcoded logic. It should
            have  ``condition`` as above and ``query``. This is not a real input, added seperately for documentation convinience.
            The object ``query`` can be one of the following

            1. ``None``: returns just ``{"input_tensor": tensor}`` dict
            2. ``'supervised'``: ``{"input_tensor": tensor, "class": tensor}``, this will fail if incorrect ``self.class_to_id``
            3. ``'unsupervised'``: ``{"input_tensor": tensor, "output_tensor": tensor}``

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

        # when you want supervised
        set(my_kewl_dataset[5:10, "supervised"].keys()) == {"input_array", "class"}

        # when you want unsupervised
        set(my_kewl_dataset[5:10, "unsupervised"].keys()) == {"input_array", "output_tensor"}
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

  def create_batches(self, batch_size, drop_last=False, seed=4) -> 'ArrowConsumer':
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
    
    return self

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
