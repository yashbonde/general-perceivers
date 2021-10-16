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

In our day to day life, we call data is nothing but an interplay of ``P`` and ``S``.

Documentation
-------------
"""

import torch


class Consumer():
    def __init__(self, fps, tokenizer, seqlen=512, batch_size=1, seed = 4):
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
              #. **(F3)** dict of categories: ``{"cat1": ["file1.txt", "file2.txt", ...], "cat2": ["file3.txt", "file4.txt", ...]}``

          tokenizer (TokenizerObject): Object of the Tokenizer
          seqlen (int, optional): Length of the Sequence. Defaults to 512.
          batch_size (int, optional): Size of the Batches. Defaults to 1.
          seed(int, optional): Seed value for randomness used to determine the batches
        """
        # parse the fps and covert to fixed internal reprensentaion -> {"meta": ["file1.txt", "file2.txt", ...]}
        if isinstance(fps, list):
            if isinstance(fps[0], str): # F0
                fps = {"null": fps}  # list of files will start with null category
            elif isinstance(fps[0], dict): # F1
                fps = {}
                for x in fps:
                    k = list(x.keys())[0]
                    v = list(x.values())[0]
                    fps.setdefault(v, []).append(k)  # list of dicts will start with category as key
            else:
                raise ValueError("fps is not in the correct format")
        elif isinstance(fps, dict):
            k = next(iter(fps))
            v = fps[k]
            assert isinstance(k, str), f"key has to be a string got: {type(k)}"
            if isinstance(v, list): # F2
                # this is the format we want so just perform checks
                assert all([isinstance(_v, list) for _k,_v in fps.items()]), "All values should be a list"
            elif isinstance(v, str): # F3
                assert all([isinstance(_v, str) for _k,_v in fps.items()]), "All values should be a string"
                fps[k] = [v]  # dict with strings as values gets converted to list of strings
        else:
            raise ValueError(f"fps is not in the correct format: {type(fps)}")
        self.fps = fps

    def __getitem__(self, x = None) -> torch.Tensor:
        """
        Args:
        
            x(Any): There is only one input since this is a special method. We take in this input item and process it accordingly based on following rules:

                #. **(I0)** ``None``: when x is None we have an internal idx that is incremented and the next batch is returned
                #. **(I1)** ``int``: when x is an int we return the batch at that index
                #. **(I2)** ``slice``: when x is a slice we return the batches in the slice
                #. **(I3)** ``list/tuple``: when x is a list we return the batches in the list containing the indices (``int``)
                #. **(I4)** ``dict -> ints``: when values of x are ints we return the batches in the list containing the indices (``int``)
                #. **(I5)** ``dict -> list``: when values of x are lists we return the batches in the list containing the indices (``list``)

        Using this is very simple. T

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
            }]                          # return random batches of 10 samples from class cat and 4 samples from class dog
            out = my_kewl_dataset[{
                "cat": [0, 1, 2, 3, 4],
                "dog": [5, 6, 7, 8, 9]
            }]                          # return the batches at indices [0...4] and [5...9] from class cat and class dog respectively
        """
        if x == None: # i0
            batch_data = self.idx_to_ds[self.__auto_idx]
            self.__auto_idx += 1
        elif isinstance(x, int): # i1
            batch_data = self.idx_to_ds[x]
        elif isinstance(x, slice): # i2
            batch_data = self.idx_to_ds[x]
        elif isinstance(x, (list, tuple)): # i3
            assert isinstance(x[0], int), f"Items in list must be integers"
            batch_data = [self.idx_to_ds[i] for i in x]
        elif isinstance(x, dict):
            assert isinstance(list(x.values())[0], (int, list)), f"Values in dict must be integers or lists"
            batch_data = []
            for k, v in x.items():
                if isinstance(v, int): # i4
                    batch_data.extend(self.class_to_idx.sample(k, v))
                elif isinstance(v, list): # i5
                    batch_data.extend([self.idx_to_ds[i] for i in v])
        else:
            raise KeyError(f"Invalid input type: {type(x)}")

