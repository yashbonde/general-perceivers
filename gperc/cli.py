r"""
CLI
===

**DO NOT USE (WIP)**

This module contains the command line interface for gperc. Is this really needed? I am not sure.
But having something along the lines of CLI means that running orchestration jobs would be possible.
Add code to your github actions by loading data from one place, dispatch this for training and then
testing it out all via CLI.

I am using ``python-fire`` from google for this, `link <https://github.com/google/python-fire>`_, that
can convert any arbitrary python object to a CLI. Normally I would use this with a ``main`` function,
but since there has to be configuration in the CLI, I am using a class ``Main`` (yes, I know, weird).

The default CLI has the following structure:

.. code-block::

    python3 -m gperc [BEHEVIOUR] [CONFIGS] [TASKS]

    BEHEVIOUR:

        -h, --help: Show this modal and exit.

        main: Run the main orchestration
        serve (WIP): serve using YoCo

    CONFIGS:

        main: configurations for the main orchestration.

            train: python3 -m gperc main train -h 
            data: python3 -m gperc main data -h
            arch: python3 -m gperc main arch -h

        serve (WIP): configurations for the server mode.

            port: python3 -m gperc serve -h
        
    TASKS:

        Tasks are specific to behaviour and can raise errors for incorrect configs

        main: tasks for the main orchestration.

            profile: python3 -m gperc main [CONFIGS] profile -h
            start: python3 -m gperc main [CONFIGS] start -h
            deploy: deploy model on NimbleBox.ai. python3 -m gperc main [CONFIGS] deploy -h


This is how something like loading a dataset and running a model would look like:

.. code-block:: bash
    
    python3 -m gperc main --modality "image/class" \
        data --dataset_name "cifar10" \
        arch --mno [1024,128,1] --ced [3,32,10] \
        train --epochs 10 --batch_size 32 --lr 0.001 \
        start

In the first line we have invoked the ``gperc`` module with modality (read below), in the next three
lines we have specified the data, the architecture and the training parameters. It ends with the
``start`` command, which starts the training.
"""


import logging
from typing import List

import torch
from torch.profiler import profile, record_function, ProfilerActivity

from .models import Perceiver
from .configs import PerceiverConfig


class Main:
    def __init__(
        self,
        mno: List,
        cde: List,
        ffw_width: float = 1.0,
        num_heads: int = 2,
        num_layers: int = 2,
        decoder_reduction: str = "mean",
        decoder_residual: bool = False,
        decoder_projection: bool = True,
        dropout: float = 0.1,
        n_classes: int = None,
        output_pos_enc: bool = False,
    ):
        r"""This is the main class for manging things from CLI. Errors are raised by the gperc.models and not here, so __setup() will
        throw errors

        Args:
            mno (List): The first dimension of input, latent and output arrays
            cde (List): The second dimension of input, latent and output arrays
            ffw_width (float, optional): The width of the feed forward layer as ratio of dims
            num_heads (int, optional): The number of attention heads
            num_layers (int, optional): The number of (latent) layers
            decoder_reduction (str, optional): After the decoder, how should the output be reduced, should be one of gperc.models.VALID_REDUCTIONS
            decoder_residual (bool, optional): Whether output array performs residual connection with the latent array
            decoder_projection (bool, optional): Is decoder output projected to a certain size
            dropout (float, optional): The dropout rate
            n_classes (int, optional): The number of classes in the output array, must be set if decoder_projection
            output_pos_enc (bool, optional): Whether to use position encoding in the decoder
        """
        config = PerceiverConfig(
            input_len=mno[0],
            input_dim=cde[0],
            latent_len=mno[1],
            latent_dim=cde[1],
            output_len=mno[2],
            output_dim=cde[2],
            ffw_latent=int(mno[1] * ffw_width),
            ffw_output=int(mno[2] * ffw_width),
            num_heads=num_heads,
            num_layers=num_layers,
            decoder_reduction=decoder_reduction,
            decoder_residual=decoder_residual,
            decoder_projection=decoder_projection,
            dropout=dropout,
            n_classes=n_classes,
            output_pos_enc=output_pos_enc,
            pos_init_std=0.02,
        )
        self._model = Perceiver(config)

    def profile(self, input_shape: List, sort_by: str = "cpu_time"):
        r"""Profile the input based on the configurations given above.

        Args:

            input_shape (List): what should be the input shape of the tensor
            sort_by (str): one of ``cpu_time, cuda_time, cpu_time_total, cuda_time_total, cpu_memory_usage, cuda_memory_usage, self_cpu_memory_usage, self_cuda_memory_usage, count``
        """
        sample_input = torch.randn(*input_shape)
        with profile(
            activities=[ProfilerActivity.CPU],
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        ) as prof:
            with record_function("gperc_inference"):
                self._model(sample_input)

        print(prof.key_averages(group_by_input_shape=True).table(sort_by=sort_by))
        prof.export_chrome_trace("trace.json")


class Serve:
    pass
