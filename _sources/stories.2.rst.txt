2. Distributed Computing
========================

This `code <https://github.com/yashbonde/general-perceivers/blob/master/distributed/pipe.py>`_ has been
tested on `NimbleBox.ai <https://nimblebox.ai/>`_ instance with two Nvidia-T4 cards.

In this section we will see how you can train 1Bn+ parameter models directly from ``gperc`` using ``GPipe``
that allows model parallelism while avoiding the bottleneck that comes with using large batches by spliting
the compute for each batch in mini-batches so devices have higher utilisation.

.. image:: assets/gpipe.png
   :alt: Alternative text

Some reference papers:

1. `torch GPipe <https://arxiv.org/pdf/2004.09910.pdf>`_: This paper extends the original
`Gpipe <https://ai.googleblog.com/2019/03/introducing-gpipe-open-source-library.html>`_ for torch.

2. The code from above paper was added into the ``pytorch`` and
`this blog <https://pytorch.org/tutorials/intermediate/pipeline_tutorial.html>`_ has tutorial for it.

Using ``GPipe``
---------------

I won't go into details of why ``GPipe`` was built and how ``pytorch`` handles it internally but rather go
over the coding decisions.

1. All the functions take in dedicated keywords in the ``forward()`` method because sending ina tuple that
   can then be split requires serious modification to the ``pytorch`` source code that can be a tiresome
   and tedious process. Example:

.. code-block:: python

   # forward method for Embeddings Module
   def forward(self, input_array, attention_mask=None, output_array=None)

2. Now the ``attention_mask`` and ``output_array`` can be ``None`` but when using ``Pipe`` pytorch consumes
   only tensors and so I have added this weird quirk where you can send in tensors with same first dimension
   with values ``-69`` for ignoring ``attention_mask`` and ``-420`` for ignoring ``output_array``. Yes very
   childish, I know. So in the script you will see code like this:

.. code-block:: python

   # output_array needs to be set None so be pass tensor with values -420
   model_input = (
      inputs.cuda(0),
      attn_mask.cuda(0),
      torch.tensor([-420. for _ in range(inputs.shape[0])]).cuda(0)
   )

3. You will need to experiment with values ``chunks`` (total number of chunks to break the input into) and
   ``partition_len`` (number of modules on each chip). Returned attentions list also is chunked ie.

.. code-block:: python

   chunks = 16; batch_size = 32
   output, attentions = model(*model_input)

   # number of attentions == number of chunks
   len(attentions) # 16

   # batch size of attention layer is batch_size / chunks
   attentions[0][0].shape[0] == 2
   

More 🍰 on the way
------------------
