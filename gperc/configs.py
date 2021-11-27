r"""
Configs
=======

``PerceiverConfig`` is the final config object that is fed to the model, but it requires knowing
exactly what you need to know about the data and the architecture. For this very purpose, there are
some simpler configs that are more convenient to use in some cases. They are:

* ``TextConfig``: A config that is used for text classification tasks.
* ``ImageConfig``: A config that is used for image tasks, supports ``classification`` and ``segmentation``.

Discussion
----------

At it's core the model processes either signals (image, audio, time-series) or it consumes discrete inputs
(tokens) that gets converted to signals by using embeddings. This simplicity and abstraction has to be
brought to ``config`` as well, currently each use case has it's own config. We can take inspiration from:

1. `PEP-518 <https://www.python.org/dev/peps/pep-0518/>`_ which talks about using TOML for ``pyproject.toml``


Documentation
-------------
"""

import json
import math
from pprint import pformat
from typing import Callable, Tuple


class PerceiverConfig:
    def __init__(
        self,
        # first set of parameters are for the architecture numbers
        input_len: int = 64,
        input_dim: int = 8,
        latent_len: int = 4,
        latent_dim: int = 16,
        output_len: int = 1,
        output_dim: int = 10,
        ffw_latent: int = 32,
        ffw_output: int = 32,
        num_heads: int = 2,
        num_layers: int = 2,
        # second set of parameters are specially for encoder and decoder behavior
        input_type: str = "raw",
        input_num_tokens: int = None,
        decoder_reduction: str = "mean",
        decoder_residual: bool = False,
        decoder_projection: bool = True,
        n_classes: int = None,
        # third set of parameters are for the initialiaations and dropouts
        pos_init_std: float = 0.02,
        dropout: float = 0.1,
        seed: int = 4,
        # user can send in kwargs if it wants to store any value
        **kwargs
    ):
        r"""Since perciever is such a powerful and versatile model, we need a good config for this.
        Different application we will simply define different configurations and wrap them in some
        model registry-kinda thing. There are many attributes in the config file and the user must
        understand what they are doing.

        I highly recommend reading `examples <stories.1.html>`__ before you start working with this.

        Args:
            input_len (int, optional): (``m``) The length of the input space
            input_dim (int, optional): (``c``) The dimension of the input space
            latent_len (int, optional): (``n``) The length of the latent space
            latent_dim (int, optional): (``d``) The dimension of the latent space
            output_len (int, optional): (``o``) The length of the output space
            output_dim (int, optional): (``e``) The dimension of the output space
            ffw_latent (int, optional): The dimension of the latent space in the feed-forward
            ffw_output (int, optional): The dimension of the output space in the feed-forward
            num_heads (int, optional): The number of heads in the multi-head attention
            num_layers (int, optional): The number of layers in the encoder and decoder
            input_type (str, optional): The type of the input space. Can be either ``raw`` or ``tokens``
            input_num_tokens (int, optional): If the ``input_type == 'tokens'`` what is the number of tokens
            decoder_reduction (str, optional): After the decoder, how should the output be reduced, should be one of
                ``"mean", "max", "sum", "min", "last", "first", None``
            decoder_residual (bool, optional): Whether ``output_array`` combines with ``latent_array``
            decoder_projection (bool, optional): Whether apply projection on ``output_array``
            n_classes (int, optional): The number of classes in the classification task, must be set if
                ``decoder_projection == True``
            pos_init_std (float, optional): The standard deviation of the position encoding
            dropout (float, optional): The dropout rate
            seed (int, optional): The seed for the random number generator
            **kwargs: Any other arguments to be stored in the config
        """

        # first set of parameters are for the architecture numbers
        self.input_len = input_len
        self.input_dim = input_dim
        self.latent_len = latent_len
        self.latent_dim = latent_dim
        self.output_len = output_len
        self.output_dim = output_dim
        self.ffw_latent = ffw_latent
        self.ffw_output = ffw_output
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.input_type = input_type
        self.input_num_tokens = input_num_tokens
        self.decoder_reduction = decoder_reduction
        self.decoder_residual = decoder_residual
        self.decoder_projection = decoder_projection
        self.n_classes = n_classes
        self.pos_init_std = pos_init_std
        self.dropout = dropout
        self.seed = seed

        for k, v in kwargs.items():
            setattr(self, k, v)

    def __repr__(self) -> str:
        return pformat(self.__dict__, indent=2, sort_dicts=True)

    def __getitem__(self, key):
        return getattr(self, key)

    def to_json(self, path = None):
        _j = json.dumps(self.__dict__, indent=2, sort_keys=True)
        if path == None:
            return _j
        with open(path, "w") as f:
            f.write(_j)

    def from_json(self, path):
        with open(path, "r") as f:
            self.__dict__ = json.load(f)


class TextConfig(PerceiverConfig):
    def __init__(self, latent_dim, vocab_size, max_len, latent_frac=0.25, ffw_ratio=1.0, **kwargs):
        r"""Config class to specially deal with the text modality cases

        Args:
            latent_dim (int): The dimension of the latent space
            vocab_size (int): The size of the vocabulary
            max_len (int): The maximum length of the input sequence
            latent_frac (float): ``latent_len`` will be this multiplied by ``max_len``
            ffw_ratio (float, optional): The ratio of the feed-forward layer in Block to input dimension
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.max_len = max_len

        self.input_len = max_len
        self.input_dim = latent_dim
        self.latent_len = int(latent_frac * max_len)
        self.latent_dim = latent_dim
        self.output_len = max_len
        self.output_dim = latent_dim
        self.ffw_latent = int(self.latent_dim * ffw_ratio)
        self.ffw_output = int(self.output_dim * ffw_ratio)
        self.input_type = "tokens"
        self.input_num_tokens = self.vocab_size
        self.decoder_reduction = None
        self.decoder_residual = True
        self.decoder_projection = True
        self.n_classes = self.vocab_size

        for k, v in kwargs.items():
            setattr(self, k, v)


class ImageConfig(PerceiverConfig):
    def __init__(
        self,
        image_shape: Tuple,
        latent_len: int,
        latent_dim: int,
        n_classes: int,
        decoder_reduction: str = "mean",
        ffw_ratio: float = 1.0,
        task: str = "classification",
        **kwargs
    ):
        r"""Config class to specially deal with the image modality cases

        Args:
            image_shape (Tuple): The shape of the image in [H, W, C]
            latent_len (int): The length of the latent space
            latent_dim (int): The dimension of the latent space
            n_classes (int): The number of classes after the output space
            decoder_reduction (str, optional): Read more in the ``PerceiverConfig`` documentation above
            ffw_ratio (float, optional): The ratio of the feed-forward layer in Block to input dimension
            task (str, optional): The task to be performed, can be one of ``classification`` and ``segmentation``
        """
        super().__init__()
        self.image_shape = image_shape
        self.task = task

        self.input_len = image_shape[0] * image_shape[1]  # image is flattened to a fix shape
        self.input_dim = image_shape[2]
        self.latent_len = latent_len
        self.latent_dim = latent_dim
        self.output_len = 1
        self.output_dim = latent_dim
        self.n_classes = n_classes
        self.ffw_latent = int(ffw_ratio * self.latent_dim)
        self.ffw_output = int(ffw_ratio * self.output_dim)
        self.input_type = "raw"
        self.decoder_reduction = decoder_reduction
        self.decoder_projection = True

        if task == "classification":
            """When performing a classification task, we do not need to query from the output_array
            meaning that there is no need for cross_attention or residual connection, but there
            needs to be a projection layer to the number of classes."""
            self.decoder_residual = False

        elif task == "segmentation":
            """When performing segmentation task, the output_array will query the latent but we
            should not use the residual connection, and we should use a projection layer to the
            number of classes. Avoiding residual connection is recommended in the paper."""
            self.decoder_residual = False
            self.output_len = image_shape[0] * image_shape[1]

        else:
            raise ValueError(f"task must be one of 'classification' or 'segmentation', got {task}")

        for k, v in kwargs.items():
            setattr(self, k, v)


class AudioConfig(PerceiverConfig):
    def __init__(
        self,
        sample_rate: int,
        duration: int,
        hop_length: int,
        num_mfcc: int,
        num_segments: int,
        num_channels: int,
        latent_len: int,
        latent_dim: int,
        n_classes: int,
        **kwargs
    ):
        r"""Config class to specially deal with the audio modality cases

        Args:
            sample_rate (int): Sampling Rate of the audio in Hertz
            duration (int): Duration of the audio in seconds
            hop_length (int): Hop-length of sliding window for FFT in number of samples
            num_mfcc (int): The number of MFCC (Mel-frequency cepstral coefficients) values considered
            num_segments (int): The number of segments the audio is divided into
            num_channels (int): The number of channels in the audio sample (mono or stereo)
            latent_len (int): The length of the latent space
            latent_dim (int): The dimension of the latent space
            n_classes (int):  The number of classes after the output space

        """

        super().__init__()
        self.samples_per_track = sample_rate * duration
        self.samples_per_segment = int(self.samples_per_track / num_segments)
        self.input_len = math.ceil(self.samples_per_segment / hop_length) * num_mfcc
        self.input_dim = num_channels
        self.latent_len = latent_len
        self.latent_dim = latent_dim
        self.output_len = 1
        self.output_dim = latent_dim
        self.n_classes = n_classes
        self.input_type = "raw"
        self.decoder_cross_attention = False
        self.decoder_residual = False
        self.decoder_projection = True

        for k, v in kwargs.items():
            setattr(self, k, v)


class BinaryConfig(PerceiverConfig):
    def __init__(
        self,
        seqlen,
        vocab_size,
        latent_dim,
        latent_frac=0.1,
        n_classes = None,
        ffw_ratio = 1.0,
        task = "classification",
        **kwargs
    ):
        """This is the config format for the binary modality

        Args:
            seqlen (int): The length of the sequence (input_array)
            vocab_size (int): The size of the vocabulary
            latent_dim (int): The dimension of the latent space
            latent_frac (float, optional): ``latent_len`` will be this multiplied by ``seqlen``
            n_classes (int, optional): The number of classes after the output space
            ffw_ratio (float, optional): The ratio of the feed-forward layer in Block to input dimension
            task (str, optional): The task to be performed, can be one of ``classification`` and None
        """
        super().__init__()

        self.input_len = seqlen
        self.input_dim = latent_dim
        self.latent_len = int(seqlen * latent_frac)
        self.latent_dim = latent_dim
        self.output_len = seqlen
        self.output_dim = latent_dim
        self.ffw_latent = int(ffw_ratio * self.latent_dim)
        self.ffw_output = int(ffw_ratio * self.output_dim)
        self.input_type = "tokens"
        self.decoder_reduction = "eot" if task == "classification" else None
        self.decoder_projection = True
        self.input_num_tokens = vocab_size

        self.decoder_cross_attention = False if task == "classification" else True
        self.decoder_residual = False if task == "classification" else True

        if task == "classification":
            assert n_classes is not None, "n_classes must be specified for classification task"
            self.n_classes = n_classes
        else:
            self.n_classes = vocab_size

        for k, v in kwargs.items():
            setattr(self, k, v)
