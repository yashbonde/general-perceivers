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
suppossed to be the general progression towards. The general idea is that the trainer should be
able to select the kind of data that it wants to use. This means that there needs to be a
structured way to represent and fetch the information. This is done as follows:

#. The input data ``F`` can be loaded in 4 different styles as given in the documentation below.
#. The fetching ``I`` can happen in 6 different styles as given in the documentation below.

I am following the same system that I have implemented in
`nbox.Parsers <https://nimbleboxai.github.io/nbox/nbox.parsers.html>`_. Here is a quick brief on
**primitives** ``P`` and **structures** ``S``:

#. ``P`` are the basic data types that are used in the data. This is the actual data \
    you want your model to process.
#. ``S`` are the structures that are used to represent the data. This is how data is organised.

In our day to day life, what we call data is nothing but an interplay of one ``P`` and many ``S``.

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

Documentation
-------------
"""

import os
import math
import json
import random

import torch
import numpy as np

from functools import lru_cache
from itertools import product

from torch.nn.functional import hardswish

from .utils import set_seed


@lru_cache()
def get_vocab(n_bytes):
    B = range(2 ** 8)
    out = product(B, repeat=n_bytes)
    vocab = {x: i for i, x in enumerate(list(out))}
    return vocab


class Consumer:
    def __init__(self, fps, n_bytes=2, seqlen=1024, verbose=False, class_to_id=None, _unittesting=False):
        r"""Consumer takes in list of files along with it's meta data and becomes a callable generator.
        When calling you can tell it what kind of data that you want. It is a full fledged data engine in itself.
        This will sit in nbox one day and thus has to be engineered in such a what that it is production grade with
        good documentation. In the nbox hierarchy it sits parallel to nbox.Model thus has to continue the following
        traits as `nbox.Parsers <https://nimbleboxai.github.io/nbox/nbox.parsers.html>`_:

        #. **primitive** that tells the actual fetching instruction
        #. **structure** should be same as the source meta data

        Args:
          fps (Any): The file paths have to be the primary index inside the lists and so filepaths "fps" can look like these:

              #. **(F0)** list of strings: ``["file1.txt", "file2.txt", ...]``
              #. **(F1)** list of dicts: ``[{"file1.txt": "cat1"}, {"file2.txt": "cat2"}, ...]``
              #. **(F2)** dict of strings: ``{"file1.txt": "cat1", "file2.txt": "cat2", ...}``
              #. **(F3)** dict of categories (IR): ``{"cat1": ["file1.txt", "file2.txt", ...], "cat2": ["file3.txt", "file4.txt", ...]}``

          n_bytes (int, optional): number of bytes that make one token, 2 is a good number.
          seqlen (int, optional): the total number of tokens for each sample
          verbose (bool, optional): if True, prints out the progress of the data
          class_to_id (dict, optional): if not None, this is a dictionary that maps the class names to the integer ids.
          _unittesting (bool): This is a private variable that is used to test the data reader. Keep at False
        """
        # parse the fps and covert to fixed internal reprensentaion -> {"meta": ["file1.txt", "file2.txt", ...]}
        self._mode = None
        if isinstance(fps, list):
            if isinstance(fps[0], str):  # F0
                fps = {"null": fps}  # list of files will start with null category
                self._mode = "F0"
            elif isinstance(fps[0], dict):  # F1
                _fps = {}
                for x in fps:
                    k = list(x.keys())[0]
                    v = list(x.values())[0]
                    _fps.setdefault(v, []).append(k)  # list of dicts will start with category as key
                fps = _fps
                self._mode = "F1"
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
                self._mode = "F2"
            elif isinstance(v, list):  # F3
                # this is the format we want so just perform checks
                assert all([isinstance(_v, list) for _k, _v in fps.items()]), "All values should be a list"
                self._mode = "F3"
        else:
            raise ValueError(f"fps is not in the correct format: {type(fps)}")
        self.fps = fps

        # values set for ops
        self.__auto_idx = 0
        self.__n_classes = len(self.fps)
        self.n_bytes = n_bytes
        self.vocab_size = int(2 ** (8 * self.n_bytes))
        self.seqlen = seqlen
        self.class_to_id = class_to_id

        self.config = {
            "seqlen": seqlen,
            "n_bytes": n_bytes,
        }

        self._unittesting = _unittesting
        if _unittesting:
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
        # - times: not relevant (TODO: @yashbonde -> Casual models)
        # - blk_size and blk_count: size of the blocks and the number of blocks
        meta = {}
        for _c, files in self.fps.items():
            _meta_for_class = {
                "filepath": [],
                "extensions": [],
                "st_size": [],
            }
            for _f in files:
                s = os.stat(_f)
                for k in _meta_for_class.keys():
                    try:
                        _meta_for_class[k].append(getattr(s, k))
                    except AttributeError:
                        pass
                _meta_for_class["filepath"].append(_f)
                ext = os.path.splitext(_f)[1]
                _meta_for_class["extensions"].append(ext)

            meta[_c] = _meta_for_class
        self.full_meta = meta

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
                    "total_samples": total_samples,
                }
            )

        # Now we create the samples by reading going over file sizes and creating the samples
        # seqlen_sample_in_bytes = seqlen * n_bytes
        # each sample is a tuple of the following items (_class, filepath, seek, read_size)
        all_samples = []
        req_size = seqlen * n_bytes
        for _c, _meta in self.full_meta.items():
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
                    _remaining_bytes_in_file = _curr_size - _curr_seek
                    _remaining_bytes_in_sample = req_size - total_bytes
                    if _remaining_bytes_in_sample > _remaining_bytes_in_file:
                        # add data for this sample and then move to next file
                        sample.append((filepath[_f_idx], _curr_seek, _curr_size))
                        _f_idx += 1
                        total_bytes += sample[-1][-1] - sample[-1][-2]
                        if _f_idx == len(filepath):
                            break
                        _curr_size = sizes[_f_idx]
                        _curr_seek = 0
                    else:
                        sample.append((filepath[_f_idx], _curr_seek, _curr_seek + _remaining_bytes_in_sample))
                        _curr_seek += _remaining_bytes_in_sample
                        total_bytes += sample[-1][-1] - sample[-1][-2]

                all_samples.append(
                    {
                        "data": sample,
                        "class": _c,
                    }
                )

                if _f_idx == len(filepath):
                    break

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

        self.__unsupervised_mode = False
        self.__supervised_mode = False
        self.__batch_mode = False

    # ----- functions for pretty printing and handling of dataset object.

    def get_dict(self):
        _d = {
            "total_samples": self._total_samples,
            "mode": self._mode,
            "n_classes": self.__n_classes,
            "n_bytes": self.n_bytes,
            "seqlen": self.seqlen,
            "vocab_size": self.vocab_size,
        }
        if hasattr(self, "batch_size"):
            _d["batch_size"] = self.batch_size
        if self.__batch_mode:
            _d["total_batches"] = len(self._batches)
        return _d

    def to_json(self):
        return json.dumps(self.get_dict(), indent=2)

    def __repr__(self):
        return f"<gperc Consumer {self.to_json()}>"

    def __len__(self):
        return self._total_samples

    def set_unsupervised_mode(self, mask_frequency=0.15):
        r"""set variables required for unsupervised query mode

        Args:

            mask_frequency (float): frequency of masking of input tensor
        """
        self.mask_frequency = mask_frequency
        self.__unsupervised_mode = True

    def set_supervised_mode(self):
        r"""set variables required for supervised query mode. Currently takes nothing."""
        self.mask_frequency = None
        self.__supervised_mode = True

    # ----- the most important function

    def __getitem__(self, x=None):
        r"""This is the heart of this code, it takes in user requests and returns the data according to it. This is slightly
        technical and so we will explain it in detail. I find similarities between databases in CRUD and datasets for machine
        learning, CRUD has amazing performance and interaction tools like SQL. Datasets in ML are more like a collection of
        data, and they are not designed to be used in friendly way. Everyone's writing their own thing there but good UX requires
        satisfying the user in some kind of formula and then let them be.

        Any SQL query has the following grammar ``SELECT [columns] FROM [table] WHERE [condition]``. This is something everyone
        understands, it's very simple. In our case ``[table] == self``, i.e. the table is the dataset itself, this is no RDMS.
        The condition is very clearly desctibed in the documentation of ``x``. But ``[columns]`` (here calling it ``query``) is
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
                if query == "supervised":
                    assert self.__supervised_mode, "case I6: supervised mode is not enabled data.set_supervised_mode()"
                elif query == "unsupervised":
                    assert self.__unsupervised_mode, "case I6: unsupervised mode is not enabled data.set_unsupervised_mode()"
                x = x[0]
            else:
                raise ValueError("case I6: tuple must have either 1 or 2 elements")

        # testing requires conditional data
        if self._unittesting:
            all_samples = list(self.fps.values())[0]
            sample_by_class = self.fps
        else:
            all_samples = self.all_samples
            sample_by_class = self.samples_by_class

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
        pad = vocab[tuple(0 for _ in range(self.n_bytes))]

        # if testing return
        if self._unittesting:
            return batch_data

        # Next we take the samples and we read the data
        def __get_one_sample(sample):
            files = sample["data"]
            _class = sample["class"]
            sample = []
            for f, s, e in files:
                with open(f, "rb") as fp:
                    fp.seek(s)
                    bytes = fp.read(e - s)
                    sample.extend(bytes)

            zip_items = []
            for i in range(self.n_bytes):
                zip_items.append(sample[i :: self.n_bytes])
            samples = list(zip(*zip_items))
            seq = [vocab[x] for x in samples]
            labels = seq.copy()  # this is exclusively for unsupervised

            if len(seq) != self.seqlen:
                seq += [pad for _ in range(self.seqlen - len(seq))]

                # -100 because torch does not calculate loss for -100 value
                labels += [-100 for _ in range(self.seqlen - len(labels))]

            return seq, labels, _class

        if isinstance(batch_data, list):
            data = [__get_one_sample(x) for x in batch_data]
            seq = [x[0] for x in data]
            labels = [x[1] for x in data]
            classes = [x[2] for x in data]
        else:
            x = __get_one_sample(batch_data)
            seq = [x[0]]
            labels = [x[1]]
            classes = [x[2]]

        out = torch.tensor(seq, dtype=torch.long)

        # now we take the data structure it according to the user's request
        _dc = {"input_array": out}
        if query == "supervised":
            if isinstance(self.class_to_id, dict):
                class_tensor = torch.tensor([self.class_to_id[x] for x in classes]).long()
            else:
                raise ValueError("class_to_id dict must be a provided")
            _dc.update({"class": class_tensor})

        elif query == "unsupervised":
            mask = np.random.uniform(0, 1, tuple(out.shape)) < self.mask_frequency
            out[torch.tensor(mask)] = pad
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
