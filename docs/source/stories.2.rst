2. Distributed Computing
========================

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
