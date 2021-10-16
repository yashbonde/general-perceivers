r"""
Configs
=======

``PerceiverConfig`` is the final config object that is fed to the model, but it requires knowing
exactly what you need to know about the data and the architecture. For this very purpose, there are
some simpler configs that are more convenient to use in some cases. They are:

* ``TextConfig``: A config that is used for text classification tasks.
* ``ImageConfig``: A config that is used for image tasks, supports ``classification`` and ``segmentation``.

Documentation
-------------
"""

from pprint import pformat
from typing import Callable, Tuple


class PerceiverConfig:
    def __init__(
        self,
        input_len: int = 64,
        input_dim: int = 8,
        latent_len: int = 4,
        latent_dim: int = 16,
        output_len: int = 1,
        output_dim: int = 10,
        ffw_latent: int = 32,
        pos_init_std: float = 0.02,
        num_heads: int = 2,
        num_layers: int = 2,
        dropout: float = 0.1,
        decoder_cross_attention: bool = False,
        decoder_residual: bool = False,
        decoder_projection: bool = True,
        output_pos_enc: bool = False,
        seed: int = 4,
        pre_processing: Callable = None,
        post_processing: Callable = None,
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

            pos_init_std (float, optional): The standard deviation of the position encoding
            num_heads (int, optional): The number of heads in the multi-head attention
            num_layers (int, optional): The number of layers in the encoder and decoder
            dropout (float, optional): The dropout rate

            decoder_cross_attention (bool, optional): Whether to use cross attention in the decoder
            decoder_residual (bool, optional): Whether ``output_array`` combines with ``latent_array``
            decoder_projection (bool, optional): Whether to use a projection layer in the decoder, used for
                classification
            output_pos_enc (bool, optional): Whether to use position encoding in the decoder

            seed (int, optional): The seed for the random number generator
            pre_processing (Callable, optional): A function that takes processes the ``input_array`` tensor
            post_processing (Callable, optional): A function that takes processes the ``output_array`` tensor

            **kwargs: Any other arguments to be stored in the config
        """

        self.input_len = input_len
        self.input_dim = input_dim
        self.latent_len = latent_len
        self.latent_dim = latent_dim
        self.output_len = output_len
        self.output_dim = output_dim
        self.ffw_latent = ffw_latent
        self.pos_init_std = pos_init_std
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout
        self.decoder_cross_attention = decoder_cross_attention
        self.decoder_residual = decoder_residual
        self.decoder_projection = decoder_projection
        self.output_pos_enc = output_pos_enc
        self.seed = seed
        self.pre_processing = pre_processing
        self.post_processing = post_processing

        for k, v in kwargs.items():
            setattr(self, k, v)

    def __repr__(self) -> str:
        return pformat(self.__dict__, indent=2, sort_dicts=True)


class TextConfig(PerceiverConfig):
    def __init__(self, latent_dim: int, vocab_size: int, max_len: int, latent_frac: float, **kwargs):
        r"""Config class to specially deal with the text modality cases

        Args:
            latent_dim (int): The dimension of the latent space
            vocab_size (int): The size of the vocabulary
            max_len (int): The maximum length of the input sequence
            latent_frac (float): ``latent_len`` will be this multiplied by ``max_len``
        """
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.max_len = max_len

        self.input_len = max_len
        self.input_dim = latent_dim
        self.latent_len = int(latent_frac * max_len)
        self.latent_dim = latent_dim
        self.output_len = max_len
        self.output_dim = latent_dim

        self.decoder_cross_attention = True
        self.decoder_residual = True
        self.decoder_projection = True
        self.n_classes = vocab_size


class ImageConfig(PerceiverConfig):
    def __init__(self, image_shape: Tuple, latent_dim: int, latent_len: float, task: str = "classification", **kwargs):
        r"""Config class to specially deal with the image modality cases

        Args:
            image_shape (Tuple): The shape of the image
            latent_dim (int): The dimension of the latent space
            latent_len (int): The length of the latent space
            task (str, optional): The task to be performed, can be one of ``classification``,
                and ``segmentation``

        """
        assert task in ["classification", "segmentation"], "task must be one of 'classification' or 'segmentation'"

        super().__init__(**kwargs)
        self.image_shape = image_shape
        self.latent_len = latent_len
        self.latent_dim = latent_dim
        self.output_len = image_shape[0]
        self.output_dim = latent_dim

        self.decoder_cross_attention = False
        self.decoder_residual = False
        self.decoder_projection = True


# class Presets:
#     perciever_tiny = PerceiverConfig()

#     def perceiver_cifar10():
#         config = PerceiverConfig()
#         config.input_len = 32 * 32
#         config.input_dim = 3
#         config.decoder_len = 1
#         config.decoder_proj = True
#         config.output_pos_enc = False
#         config.decoder_residual = False
#         return config

#     perciever = PerceiverConfig(
#         pre_processing=None,
#         input_len=2048,
#         input_dim=768,
#         num_layers=26,
#         latent_len=256,
#         latent_dim=1280,
#         ffw_latent=1280,
#         output_len=2048,
#         output_dim=768,
#         ffw_output=768,
#         decoder_residual=False,
#         post_processing=None,
#     )

#     perciever_large = PerceiverConfig(
#         pre_processing=None,
#         input_len=2048,
#         input_dim=768,
#         num_layers=40,
#         latent_len=256,
#         latent_dim=1536,
#         ffw_latent=1536,
#         output_len=512,
#         output_dim=768,
#         ffw_output=768,
#         decoder_residual=False,
#         post_processing=None,
#     )
