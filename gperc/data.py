"""
Data Reader
===========

This file has the datareaders for the ``gperc`` program. The datareaders work as follows:

#. The default datareader is ``gperc.BinaryConsumer`` and it reads all the files that are \
    provided to this. It reads the binaries of the files and reads the data based on pretty \
    much the size of the file.
#. You can provide extra metadata to the datareaders. This is done by providing a dictionary \
    or list. For more information read below.

Though there is some code added below, it does not work. I have added it here because that is
suppossed to be the general progression towards.

Raw Bytes Tokenization
----------------------

I am choosing to read binary instead of tokenizing text, this is similar to how computer programs
like to work.

1. A good way to measure the amount of information processed per sample is \
    ``bytes_processed = n_bytes * seqlen``, eg. ``8192 * 2 = 16KB``. ``n_bytes`` defines the total \
    number of tokens as ``n_tokens = 2 ** (nbytes * 8) => (256, 65536, 16777216, ...)``, which is \
        total number of permutations with 2 bits.

2. This is should not be confused with memory footprint since that is going to be larger as each \
    ``int`` is 64-bits (``i64`` in Rust).

3. In the above sample with ``batch_size = 20``, we have processed ``320KB`` same as the total \
    cache on `Apple M1 <https://en.wikipedia.org/wiki/Apple_M1#CPU>`_, which has 192 KB of L1 \
    instruction cache and 128 KB of L1 data cache.

4. Total tokens processed would be ``20 * 8192 = 163840`` in each batch and with ``i64`` that means \
    memory footprint of ``163840 * 64 ~ 1.25MB``.

5. Wrapping up that means we are processing 320KB of data in a 1.25MB memory footprint (which is a \
    ``4x`` memory requirement).

Internal Representation
-----------------------

This is what we have we have to do, ``full_meta`` is a not a good way access individual elements
in the batch, so we need to convert it to a more convenient internal representation. Consider
``full_meta`` like a table so this is what it would look like:

.. list-table:: Full Meta as a Table
   :header-rows: 1

   * - class
     - filepath
     - size (in bytes)
   * - cat
     - f1
     - 137
   * - cat
     - f2
     - 417
   * - cat
     - f3
     - 139
   * - dog
     - f4
     - 123
   * - dog
     - f5
     - 52
   * - dog
     - f6
     - 390

The batches with ``seqlen = 128`` and ``n_bytes=1`` would look like a flat array with items like this:

.. code-block:: python
    
    batches = [
        ([f1,   0, 128],),
        ([f1, 128, 137],
         [f2,   0, 119],),
        ([f2, 119, 347],),
        ([f2, 347, 417],
         [f3,   0,  58],),
        ...
    ]


System Footprint
----------------

``gperc.Consumer`` is a powerful data ingestion system, and as a side effect will create new files
on your system, in order to avoid 


Documentation
-------------
"""

import os
import math
import json
import random
import hashlib

import torch
import numpy as np
import subprocess
from tqdm.auto import trange
from pprint import pprint as peepee

from .arrow import get_vocab, convert_to_gperc_consumer_ir
from .utils import set_seed


def get_time():
    import datetime
    return '{0:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now())


def binary_sample_generator(meta, seqlen, n_bytes):
    """This function takes in the filesystem metadata generated by the main ``Consumer`` class, ``seqlen``,
    ``n_bytes`` and returns all the samples in the dataset. Each sample looks like this:

    .. code-block::python

        sample = {
            "data": [
                (filepath_0, start_byte, end_byte),
                (filepath_0, start_byte, end_byte),  # in some case there are more than one files in a sample
            ],
            "class": "string"
        }
    
    The logic of the code is as follows: For each class data go over the files. For each file check the total
    number of bytes in the file. keep adding the above tuple while the total number of bytes < seqlen. At the
    end of each file we increment the current buffer +1 to account for "<EOF>" tag.
    """
    all_samples = []
    req_size = seqlen * n_bytes
    # peepee(meta)
    for _c, _meta in meta.items():
        # ----- for each label
        sizes = _meta["st_size"]
        filepath = _meta["filepath"]
        _f_idx = 0
        _curr_size = sizes[_f_idx]
        _curr_seek = 0
        while 1:
            sample = []
            total_bytes = 0
            while total_bytes < req_size:
                this_filepath = filepath[_f_idx]
                bytes_remaining_in_this_file = _curr_size - _curr_seek
                bytes_required = req_size - total_bytes

                if bytes_remaining_in_this_file > bytes_required:
                    # in this case the data in this file is more than required and so this is a simple addition
                    sample.append((this_filepath, _curr_seek, _curr_seek + bytes_required))
                    _curr_seek = _curr_seek + bytes_required
                    total_bytes += bytes_required
                else:
                    # in this case read till what is available in the file and the increment the file counter
                    sample.append((this_filepath, _curr_seek, _curr_seek + bytes_remaining_in_this_file))
                    _curr_seek = 0
                    _f_idx += 1
                    total_bytes += bytes_remaining_in_this_file

                if _f_idx == len(filepath):
                    # there are no more files in this class
                    break

                # done here to avoid the index error
                _curr_size = sizes[_f_idx]

            all_samples.append({
                "data": sample,
                "class": _c,
            })

            if _f_idx == len(filepath):
                break
    
    return all_samples


def diff_sample_generator(meta):
    """This function takes in the filesystem metadata generated by the main ``Consumer`` class, ``seqlen``,
    ``n_bytes`` and returns all the samples in the dataset. Each sample looks like this:

    .. code-block::python

        sample = {
            "data": [
                (filepath_0, start_byte, end_byte), # there will be one files in this sample
            ],
            "class": "string"
        }
    
    The logic of the code is as follows: For each class data go over the files. For each file check the total
    number of bytes in the file. keep adding the above tuple while the total number of bytes < seqlen. At the
    end of each file we increment the current buffer +1 to account for "<EOF>" tag.
    """
    all_samples = []
    for _c, _meta in meta.items():
        # ----- for each label
        sizes = _meta["st_size"]
        filepath = _meta["filepath"]
        for _f_idx, _fp in enumerate(filepath):
            all_samples.append({
                "data": [(_fp, 0, sizes[_f_idx])],
                "class": _c,
            })

    return all_samples


def decode_ids(ids, vocab):
    """Decode the ids to the corresponding characters.
    """
    out = []
    for _id in ids:
        out.append(vocab[_id])
    return out






class Consumer:
    def __init__(
        self,
        fps,
        style="diff",
        n_bytes=2,
        seqlen="auto",
        verbose=False,
        class_to_id=None,
        _unittesting=False
    ):
        r"""

        Args:
          fps (Any): The file paths have to be the primary index inside the lists and so filepaths "fps" can look like these:

              #. **(F0)** list of strings: ``["file1.txt", "file2.txt", ...]``
              #. **(F1)** list of dicts: ``[{"file1.txt": "cat1"}, {"file2.txt": "cat2"}, ...]``
              #. **(F2)** dict of strings: ``{"file1.txt": "cat1", "file2.txt": "cat2", ...}``
              #. **(F3)** dict of categories (IR): ``{"cat1": ["file1.txt", "file2.txt", ...], "cat2": ["file3.txt", "file4.txt", ...]}``

          style (str, optional): The style of the merging the data should be one of the following:

                #. **concat**: the data is threating like a very long sequence of bytes, in this case bytes are split by ``<EOF>``
                #. **diff**: each file is treated as an independent sequence of bytes, in this case bytes are padded by ``<EOF>``

          n_bytes (int, optional): number of bytes that make one token, 2 is a good number.
          seqlen (list, optional): the total number of tokens for each sample
          verbose (bool, optional): if True, prints out the progress of the data
          class_to_id (dict, optional): if not None, this is a dictionary that maps the class names to the integer ids.
          _unittesting (bool): This is a private variable that is used to test the data reader. Keep at False
        """
        # parse the fps and covert to fixed internal reprensentaion -> {"meta": ["file1.txt", "file2.txt", ...]}
        print(f"[{get_time()}] Starting IR generation")
        self.fps, self._mode = convert_to_gperc_consumer_ir(fps)
        _hash = hashlib.sha256(json.dumps(fps).encode()).hexdigest()

        # check the style
        assert style in ["concat", "diff"], f"style should be one of 'concat' or 'diff' got: {style}"

        # check the style
        assert style in ["concat", "diff"], f"style should be one of 'concat' or 'diff' got: {style}"

        # values set for ops
        self.__auto_idx = 0
        self.__n_classes = len(self.fps)
        self.style = style
        self.n_bytes = n_bytes
        self.seqlen = seqlen
        self.class_to_id = class_to_id

        # vocabulary building process special tokens
        vocab_size = int(2 ** (8 * n_bytes))
        self.EOF_ID = vocab_size
        self.EOF_TOKEN = "<EOF>"
        self.vocab_size = vocab_size + 1

        self.config = {
            "seqlen": seqlen,
            "n_bytes": n_bytes,
        }

        self._unittesting = _unittesting
        if _unittesting:
            all_samples = list(self.fps.values())
            if self._mode != "F2":
                all_samples = all_samples[0]
            self.all_samples = all_samples
            self._total_samples = len(self.all_samples)
            return

        # Before we get to capturing the metadata for each file it is important to understand
        # what is the data that we are going to get from the file. So the meta is obtained from
        # UNIX's sys/stat.h struct stat.
        # read more: https://pubs.opengroup.org/onlinepubs/000095399/basedefs/sys/stat.h.html
        # We are capturing the following:
        # - filepath: the path to the file
        # - extension: the extension of the file
        # - st_size: the size of the file in bytes
        # Other things that I was previously capturing but later removed:
        # - st_dev: Device ID of device containing file, st_ino: File serial number
        # (st_dev, st_ino) is unique to each file and thus can be used to identify the file
        # - times: not relevant (TODO: @yashbonde -> Causal models)
        # - blk_size and blk_count: size of the blocks and the number of blocks
        # 
        # The total number of tokens will be incremented by total number of files in each
        # class because of "<EOF>" logic.
        print(f"[{get_time()}] Starting Meta Creation")
        meta = {}
        print(f"[{get_time()}] Creating metadata")
        for _c, files in self.fps.items():
            _meta_for_class = {
                "filepath": [],
                "extensions": [],
                "st_size": [],
            }
            pbar = trange(len(files))

            # this is a much faster method of getting the meta data instead of manually opening many
            # different subprocesses
            fmetas = [
                len(bytearray(x.encode("utf-8"))) + 1
                for x in subprocess.check_output(["file", *files]).decode("utf-8").split("\n")
            ] # +1 to compensate for \n when splitting
            for _, _f, _fm in zip(pbar, files, fmetas):
                _meta_for_class["st_size"].append(
                    _fm + os.stat(_f).st_size + 1  # +1 for EOF
                )
                _meta_for_class["filepath"].append(_f)
                _meta_for_class["extensions"].append(os.path.splitext(_f)[1])

            meta[_c] = _meta_for_class
        self.full_meta = meta

        # peepee(self.full_meta)

        if seqlen == "auto":
            seqlen = max([max(v["st_size"]) for k,v in self.full_meta.items()])
            self.seqlen = seqlen

        # Next we must create a samples of data and create the one true location of each item in the
        # batch. This will help up locate and read things faster.
        for _c, _meta in self.full_meta.items():
            _cumm_sizes = np.cumsum(_meta["st_size"]).tolist()
            total_size = _cumm_sizes[-1]
            total_tokens = math.ceil(total_size / n_bytes)
            total_samples = math.ceil(total_tokens / seqlen)
            meta[_c].update(
                {
                    "cummulative": _cumm_sizes,
                    "total_tokens": total_tokens,
                    "total_samples": len(_meta["st_size"]) if style == "diff" else total_samples,
                }
            )
        
        # there can be a bunch of different methods that can se used to create a batch.
        # here I have added one for the classical binary sampling method.
        print(f"[{get_time()}] Create sample indices")
        if style == "concat":
            all_samples = binary_sample_generator(meta, seqlen, n_bytes)
        elif style == "diff":
            all_samples = diff_sample_generator(meta)
        self.all_samples = all_samples

        # For ensuring guarantees, we need to check if the total number of samples is same as from
        # the meta.
        assert sum([_meta["total_samples"] for _, _meta in meta.items()]) == len(
            self.all_samples
        ), "total samples through both approaches are not the same: got {} and {}".format(
            sum([_meta["total_samples"] for _, _meta in meta.items()]), len(self.all_samples)
        )
        self._total_samples = len(self.all_samples)
        self.samples_by_class = {}
        for x in self.all_samples:
            self.samples_by_class.setdefault(x["class"], []).append(x)

        # finally the dataset is ready, if verbose print the stats
        if verbose:
            print("=" * 50)
            print("Loading complete:", len(self.all_samples))
            print("Breakdown by class:")
            for k, v in self.samples_by_class.items():
                print(f"  {k}: {len(v)} ({len(v)/self._total_samples * 100:.3f}%)")
            print("=" * 50)

        self.__data_mode = None
        self.__batch_mode = False
        print(f"[{get_time()}] Consumer creation done")

    # ----- functions for pretty printing and handling of dataset object.

    def get_dict(self):
        _d = {
            "mode": self._mode,
            "n_classes": self.__n_classes,
            "n_bytes": self.n_bytes,
            "seqlen": self.seqlen,
            "vocab_size": self.vocab_size,
            "style": self.style,
        }

        upsure_items = {
            "total_samples": "_total_samples",
            "data_mode": "__data_mode",
            "batch_size": "batch_size"
        }
        for k, v in upsure_items.items():
            try:
                _d[k] = getattr(self, v)
            except:
                pass
        return _d

    def to_json(self, fp = None):
        _j = json.dumps(self.get_dict(), indent=2)
        if fp == None:
            return _j
        else:
            with open(fp, "w") as f:
                f.write(_j)

    def __repr__(self):
        return f"<gperc Consumer {self.to_json()}>"

    def __len__(self):
        return self._total_samples

    def set_unsupervised_mode(self, mask_frequency=0.15, add_cls = False):
        r"""set variables required for unsupervised query mode

        Args:
            mask_frequency (float): frequency of masking of input tensor
            add_cls (bool): whether to prefix the ``<CLS>`` token to data

        """
        self.mask_frequency = mask_frequency
        self.__data_mode = "unsupervised"

    def set_supervised_mode(self):
        r"""set variables required for supervised query mode. Currently takes nothing."""
        assert isinstance(self.class_to_id, dict), f"class_to_id must be a dict, got: {type(self.class_to_id)}"
        self.__data_mode = "supervised"
        self.mask_frequency = None
        self.__data_mode = "supervised"

    # ----- the most important function

    def __getitem__(self, x=None):
        
        # check if case I6 and create necessary variables
        query = None
        if isinstance(x, tuple):
            if len(x) == 1:
                x = x[0]
            elif len(x) == 2:
                assert x[1] in [
                    "supervised",
                    "unsupervised",
                    None,
                ], "case I6: second argument must be None or 'supervised' or 'unsupervised'"
                query = x[1]
                assert query == self.__data_mode, f"case I6: query ({query}) must be same as data mode ({self.__data_mode})"
                x = x[0]
            else:
                raise ValueError("case I6: tuple must have either 1 or 2 elements")

        # testing requires conditional data
        sample_by_class = self.fps if self._unittesting else self.samples_by_class
        all_samples = self.all_samples

        # fetching based on a bunch of different indexing methods
        if x == None:  # i0
            batch_data = all_samples[self.__auto_idx]
            self.__auto_idx += 1

        elif isinstance(x, int):  # i1
            batch_data = all_samples[x]

        elif isinstance(x, slice):  # i2
            batch_data = all_samples[x]

        elif isinstance(x, (list, tuple)):  # i3
            assert isinstance(x[0], int), f"Items in list must be integers"
            batch_data = [all_samples[i] for i in x]

        elif isinstance(x, dict):
            if len(self.fps) == 1 and "null" in self.fps:
                raise ValueError("There is no category provided, so you cannot try to make a batch from a dict")
            assert isinstance(list(x.values())[0], (int, list)), f"Values in dict must be integers or lists"
            keys_in_x_not_in_fps = set(x.keys()).difference(set(self.fps.keys()))
            assert keys_in_x_not_in_fps == set(), f"Keys in dict must be in fps: {keys_in_x_not_in_fps}"
            batch_data = []
            for k, v in x.items():
                samples = sample_by_class[k]
                if isinstance(v, int):  # i4
                    assert v > 0, f"Values in dict must be positive integers"
                    batch_data.extend(np.random.choice(samples, v, replace=False).tolist())
                elif isinstance(v, list):  # i5
                    batch_data.extend([samples[i] for i in v])
        else:
            raise KeyError(f"Invalid input type: {type(x)}")

        vocab = get_vocab(self.n_bytes)

        # if testing return
        if self._unittesting:
            return batch_data

        # Next we take the samples and we read the data
        def __get_one_sample(sample):
            files = sample["data"]
            _class = sample["class"]
            sample = []
            for i, (f, s, e) in enumerate(files):
                sample.extend(subprocess.check_output(["file", f]))
                e = e - self.n_bytes
                with open(f, "rb") as fp:
                    fp.seek(s)
                    bytes = fp.read(e - s)
                    sample.extend(bytes)

            zip_items = [sample[i :: self.n_bytes] for i in range(self.n_bytes)]
            samples = list(zip(*zip_items))
            seq = [vocab[x] for x in samples] + [self.EOF_ID,]
            attention_mask = [1,] * len(seq)

            labels = seq.copy()  # this is exclusively for unsupervised

            if len(seq) < self.seqlen:
                seq += [self.EOF_ID for _ in range(self.seqlen - len(seq))]
                attention_mask += [0,] * (self.seqlen - len(attention_mask))

                # -100 because torch does not calculate loss for -100 value
                labels += [-100 for _ in range(self.seqlen - len(labels))]

            if len(seq) > self.seqlen:
                seq = seq[:self.seqlen]
                attention_mask = attention_mask[:self.seqlen]
                labels = labels[:self.seqlen]

            return seq, labels, _class, attention_mask

        if isinstance(batch_data, list):
            data = [__get_one_sample(x) for x in batch_data]
            seq = [x[0] for x in data]
            labels = [x[1] for x in data]
            classes = [x[2] for x in data]
            attention_mask = [x[3] for x in data]
        else:
            x = __get_one_sample(batch_data)
            seq = [x[0]]
            labels = [x[1]]
            classes = [x[2]]
            attention_mask = [x[3]]

        out = torch.tensor(seq, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)

        # now we take the data structure it according to the user's request
        _dc = {"input_array": out, "attention_mask": attention_mask}
        if query == "supervised":
            if isinstance(self.class_to_id, dict):
                class_tensor = torch.tensor([self.class_to_id[x] for x in classes]).long()
            else:
                raise ValueError("class_to_id dict must be a provided")
            _dc.update({"class": class_tensor})

        elif query == "unsupervised":
            mask = np.random.uniform(0, 1, tuple(out.shape)) < self.mask_frequency
            out[torch.tensor(mask)] = self.EOF_ID
            labels = torch.tensor(labels, dtype=torch.long)
            _dc.update({"input_array": out, "output_array": labels})

        return _dc

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

    def get_next_batch(self, query=None):
        assert self.__batch_mode, "You must create batches first data.create_batches()"
        try:
            x = next(self._iter_batches)
        except StopIteration:
            self.create_batches(self.batch_size, self.drop_last, self.seed + 1)
            x = next(self._iter_batches)

        return self[x, query]

