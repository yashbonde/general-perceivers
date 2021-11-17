r"""
Perceiver Model
===============

This file has code on the neural network of the pervceiver architecture. ``gperc.models.Perceiver`` sits at
the heart of this project. Use ``Perceiver`` for `everyday <stories.1.html>`_ use of the model, when you want
to train really large models with model parallellism read `here <stories.2.html>`_.


Distributed
-----------

``gperc`` out-of-box can handle distributed model parallel training with
`get_distributed_model() <gperc.models.html#gperc.models.get_distributed_model>`_ During distributed training
and inference with ``torch.distributed.pipeline.sync.Pipe``
(`read tutorial <https://pytorch.org/tutorials/intermediate/pipeline_tutorial.html>`_) the input has to be a
``nn.Sequential`` object.

Documentation
-------------
"""

import math
import torch
from torch import nn
from torch.nn import functional as F

VALID_REDUCTIONS = ["mean", "max", "sum", "last", "first", "eot", None]


def build_position_encoding(position_encoding_type, config, num_index_items, emb_dim):
    r"""Get the positional encoding matrix. If ``position_encoding_type == "trainable"`` then a random normal
    matrix is returned, if it is "sinusoid" then

    Args:
        position_encoding_type (str): type of embedding, should be one of "trainable", "sinusoid"
        config: ``gperc.PerceiverConfig``
        num_index_items (int): number of items in the embedding, eg. ``vocab_size``
        emb_dim (int): embedding dimension

    Returns:
        ``torch.nn.Parameter``: Item that can be used as a parameter in a ``torch.nn.Embedding``
    """
    if position_encoding_type == "trainable":
        # learnable positional embedding is nothing but a random normal matrix
        latent_pos_emb = nn.Parameter(torch.normal(mean=0.0, std=config.pos_init_std, size=(num_index_items, emb_dim)))
        return latent_pos_emb
    elif position_encoding_type == "sinusoid":
        # sinusoid positional embedding is same as from the paper

        pe = torch.zeros(num_index_items, emb_dim)
        position = torch.arange(0, num_index_items, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, emb_dim, 2).float() * (-math.log(10000.0) / emb_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        # grad is false because this cannot be learnable
        return nn.Parameter(pe, requires_grad=False)


# ============= self attention blocks =============== #


class Block(nn.Module):
    def __init__(self, kv_dim, q_dim, num_heads, ffw_dim, dropout=0.0, add_residual=False):
        r"""Generic block with Attention and MLP layers

        Args:
            kv_dim (int): dimension of the key-value embeddings
            q_dim (int): dimension of the query embeddings
            num_heads (int): number of heads in the multihead attention
            ffw_dim (int): dimension of the feed-forward layer
            dropout (float, optional): dropout rate
            add_residual (bool, optional): whether to add residual to the query
        """
        super().__init__()
        assert q_dim % num_heads == 0, "Latent Dimension must be divisible by number of heads"

        self.kv_dim = kv_dim
        self.q_dim = q_dim
        self.dim = q_dim
        self.num_heads = num_heads
        self.ffw_dim = ffw_dim
        self.add_residual = add_residual

        # layer norm the inputs
        self.lnkv = nn.LayerNorm(kv_dim)
        self.lnq = nn.LayerNorm(q_dim)

        # items for attention
        self.fv = nn.Linear(kv_dim, q_dim)
        self.fk = nn.Linear(kv_dim, q_dim)
        self.fq = nn.Linear(q_dim, q_dim)
        self.drop_att = nn.Dropout(dropout)
        self.fo = nn.Linear(q_dim, q_dim)

        # items for mlp
        self.lnqkv = nn.LayerNorm(q_dim)
        self.mlp = nn.Sequential(
            nn.Linear(q_dim, ffw_dim),
            nn.GELU(),
            nn.Linear(ffw_dim, q_dim),
        )
        self.drop_mlp = nn.Dropout(dropout)

    def forward(self, kv, q, attn_mask=None):
        r"""Forward pass of the block that taken in a a key-value tensor and a query tensor and performs
        the attention and mlp layers. Since it consumes ``kv`` and ``q`` seperately, the blocks are responisble
        for cross attention like features. Returns a

        Args:
            kv (torch.Tensor): tensor to extract information from
            q (torch.Tensor): tensor for querying the information

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: tuple of output Tensor and Attention matrix
        """
        # first layer norm the inputs
        _q = self.lnq(q)
        _kv = self.lnkv(kv)

        # then compute the query, key, value and split for multihead attention
        Q, K, V = self.fq(_q), self.fk(_kv), self.fv(_kv)
        Q, K, V = tuple(map(lambda x: x.view(x.shape[0], self.num_heads, -1, x.shape[-1] // self.num_heads), (Q, K, V)))
        A = Q @ K.permute(0, 1, 3, 2) * (self.dim ** -0.5)  # [b, h, n, e/h] @ [b, h, e/h, m] -> [b, h, n, m]

        # though the logic below can be simplified, it is kept for debugging purposes
        if attn_mask != None:
            """
            An atttention mask (attn_mask) looking like this ===>

            [[[[    -0.,     -0.,     -0., -10000., -10000., -10000.]]],
             [[[    -0.,     -0.,     -0., -10000., -10000., -10000.]]],
             [[[    -0.,     -0.,     -0.,     -0., -10000., -10000.]]],
             [[[    -0.,     -0.,     -0., -10000., -10000., -10000.]]]])

             The attention (A) looking like this ===>

            [[[[ 0.0912,  0.3802,  0.3254,  0.2963,  0.2943,  0.3186],
               [-0.0831,  0.0890,  0.1677,  0.4240,  0.4294,  0.4220],
               [ 0.2721, -0.1913,  0.0315,  0.1929,  0.1970,  0.2083]]],
             [[[ 0.3262,  0.4335, -0.0190,  0.2442,  0.3023,  0.3050],
               [ 0.3434, -0.1238,  0.1008,  0.4174,  0.4255,  0.4295],
               [ 0.2231, -0.0568,  0.3779,  0.2310,  0.2054,  0.2230]]],
             [[[-0.3235,  0.0440, -0.1690, -0.1334,  0.3116,  0.1234],
               [ 0.2854,  0.3651,  0.0088,  0.2551,  0.4243,  0.1909],
               [ 0.0854,  0.1475,  0.0654,  0.1174,  0.2129,  0.1210]]],
             [[[ 0.1793, -0.2151,  0.0015,  0.2963,  0.3001,  0.0604],
               [ 0.0511, -0.0491,  0.0312,  0.4240,  0.4306,  0.2473],
               [ 0.2740,  0.4460,  0.1878,  0.1929,  0.2253,  0.1717]]]],

            Should given the result after the attention mask + softmax is applied ===>

            [[[[0.2778, 0.3710, 0.3512, 0.0000, 0.0000, 0.0000],
               [0.2880, 0.3420, 0.3700, 0.0000, 0.0000, 0.0000],
               [0.4140, 0.2605, 0.3255, 0.0000, 0.0000, 0.0000]]],
             [[[0.3544, 0.3946, 0.2510, 0.0000, 0.0000, 0.0000],
               [0.4147, 0.2599, 0.3254, 0.0000, 0.0000, 0.0000],
               [0.3421, 0.2586, 0.3994, 0.0000, 0.0000, 0.0000]]],
             [[[0.2074, 0.2996, 0.2421, 0.2509, 0.0000, 0.0000],
               [0.2624, 0.2841, 0.1990, 0.2545, 0.0000, 0.0000],
               [0.2453, 0.2610, 0.2404, 0.2533, 0.0000, 0.0000]]],
             [[[0.3982, 0.2684, 0.3334, 0.0000, 0.0000, 0.0000],
               [0.3466, 0.3136, 0.3398, 0.0000, 0.0000, 0.0000],
               [0.3221, 0.3825, 0.2954, 0.0000, 0.0000, 0.0000]]]]
            """
            A = A + attn_mask
        A = A.softmax(dim=-1)  # [b, h, n, m]

        A = self.drop_att(A)
        out = (A @ V).reshape((q.shape[0], -1, self.q_dim))  # [b, h, n, m] @ [b, h, m, e/h] -> [b, h, n, e/h] -> [b, n, e]
        out = self.fo(out)

        # Optionally include a residual to the query: Consider omitting the residual if the semantics of query and output
        # are different, e.g. if queries are positions and outputs are pixels.
        if self.add_residual:
            out = out + q

        # now we will pass it through the mlp
        out = self.mlp(self.lnqkv(out)) + out
        out = self.drop_mlp(out)

        return out, A


# ============= layer blocks =============== #


class Embeddings(nn.Module):
    def __init__(self, config):
        def __check_conditionals():
            assert config.input_type in ["raw", "tokens"], "input_type should be one of ['raw', 'tokens']"
            if config.input_type == "tokens":
                assert config.input_num_tokens > 0, "when input_type is 'tokens', input_num_tokens should be > 0"

        __check_conditionals()
        super().__init__()
        self.config = config

        if config.input_type == "tokens":
            self.input_embedding = nn.Embedding(config.input_num_tokens, config.input_dim)

        self.pos_emb_encoder = build_position_encoding("trainable", config, config.input_len, config.input_dim)
        self.pos_emb_processor = build_position_encoding("trainable", config, config.latent_len, config.latent_dim)
        self.pos_emb_decoder = build_position_encoding("trainable", config, config.output_len, config.output_dim)

    def forward(self, input_array, attention_mask=None, output_array=None):
        r"""Takes in either the ``input_array`` or tuple with 3 items ``(input_array, attention_mask, output)``
        and returns a tuple with 4 values ``(input_array, attention_mask, latent_array, output_array)``. If configured
        ``input_array`` can have tokens and will be automatically embedded.

        .. note::

            When using GPipe you need to send in tensors because it will try to send items as microbatches
            for each GPU. Now that requires all the inputs to be tensors, so here I have written some
            basic dumb heuristic that can set attention_mask and output_array to None if average of the values
            in those tensors is -69 and -420 resp.

            Image classification task does not require any ``attention_mask`` you can pass that as a tensor
            with values ``attention_mask = torch.tensor([-69. for _ in range(batch_size)])`` and similarly you
            can send ``output_array`` as a tensor with values ``output_array = torch.tensor([-420. for _ in range(batch_size)])``

        """

        if attention_mask != None:
            if attention_mask.sum() / torch.numel(attention_mask) == -69.0:
                attention_mask = None
            else:
                assert (
                    attention_mask.shape == input_array.shape[:2]
                ), f"mask shape ({attention_mask.shape}) != input shape ({input_array.shape[:2]})"

        if output_array != None:
            if output_array.sum() / torch.numel(output_array) == -420.0:
                output_array = None

        dtype = self.pos_emb_decoder.data.dtype

        # if the input is tokens then apply the embedding and add positional encoding
        if hasattr(self, "input_embedding"):
            input_array = self.input_embedding(input_array)
        input_array = input_array + self.pos_emb_encoder[None, : input_array.size(1), ...]

        if attention_mask is not None:
            attention_mask = attention_mask[:, None, None, :]
            attention_mask = attention_mask.to(dtype=dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * -10000.0

        # if user has apready provided output_array then use that and add positional encoding
        if output_array is None:
            output_array = torch.cat([self.pos_emb_decoder[None, ...] for _ in range(input_array.shape[0])], dim=0)
        if output_array is not None:
            output_array = output_array + self.pos_emb_decoder[None, : output_array.size(1), ...]

        # latent array by deafault is always going to be the positional encoding
        latent_array = torch.cat([self.pos_emb_processor[None, ...] for _ in range(input_array.shape[0])], dim=0)

        # reeturn a tuple, because this block will be used in nn.Sequential
        return input_array, attention_mask, latent_array, output_array


class EncoderBlock(nn.Module):
    def __init__(self, config):
        r"""Encoder Block with postional embeddings"""
        super().__init__()
        self.config = config
        self.b = Block(
            kv_dim=config.input_dim,
            q_dim=config.latent_dim,
            num_heads=config.num_heads,
            ffw_dim=config.ffw_latent,
            dropout=config.dropout,
            add_residual=True,
        )

        self.drop = nn.Dropout(config.dropout)

    def forward(self, input_array, attention_mask, latent_array, output_array):
        r"""takes in a tuple with 4 values ``(input_array, attention_mask, latent_array, output_array)``
        and returns a tuple with 3 items ``(latent_array, output_array, attentions)``"""
        input_array = self.drop(input_array)
        latents, A = self.b(input_array, latent_array, attention_mask)
        return latents, output_array, [A]


class ProcessorBlock(nn.Module):
    def __init__(self, config):
        r"""Processor Block without positional embeddings"""
        super().__init__()
        self.config = config
        self.b = Block(
            kv_dim=config.latent_dim,
            q_dim=config.latent_dim,
            num_heads=config.num_heads,
            ffw_dim=config.ffw_latent,
            dropout=config.dropout,
            add_residual=True,
        )

    def forward(self, latent_array, output_array, attentions):
        r"""takes in a tuple with 3 values ``(latent_array, output_array, attentions)``
        and returns a tuple with 3 items ``(latent_array, output_array, attentions)``"""
        # latent_array, output_array, attentions = x
        latents, A = self.b(latent_array, latent_array)
        attentions.append(A)
        return latents, output_array, attentions


class DecoderBlock(nn.Module):
    def __init__(self, config):
        def __check_conditionals():
            if config.decoder_projection:
                assert hasattr(config, "n_classes"), "Must have n_classes > 0 if using projection"
            if config.n_classes:
                assert config.decoder_projection, "Must have decoder_projection if has n_classes"
            assert (
                config.decoder_reduction in VALID_REDUCTIONS
            ), f"decoder_reduction must be in {VALID_REDUCTIONS} , got '{config.decoder_reduction}'"

        __check_conditionals()

        super().__init__()
        self.config = config
        self.b = Block(
            kv_dim=config.latent_dim,
            q_dim=config.output_dim,
            num_heads=config.num_heads,
            ffw_dim=config.ffw_latent,
            dropout=config.dropout,
            add_residual=config.decoder_residual,
        )

        if config.decoder_projection:
            self.projection = nn.Linear(config.output_dim, config.n_classes)

    def forward(self, input_array, latent_array, output_array, attentions):
        r"""takes in a tuple with 3 values ``(latent_array, output_array, attentions)``
        and returns a tuple with 2 items ``(output_logits, attentions)``"""
        # latent_array, output_array, attentions = x
        out, A = self.b(latent_array, output_array)
        attentions.append(A)

        # aggregate the decoder output across length
        if self.config.decoder_reduction == "mean":
            out = out.mean(dim=1)
        elif self.config.decoder_reduction == "max":
            out = F.max_pool1d(out.permute(0, 2, 1), out.shape[1], None, 0, 1, False, False).permute(0, 2, 1).squeeze(1)
        elif self.config.decoder_reduction == "sum":
            out = out.sum(dim=1)
        elif self.config.decoder_reduction == "last":
            out = out[:, -1, :]
        elif self.config.decoder_reduction == "first":
            out = out[:, 0, :]
        elif self.config.decoder_reduction == "eot":
            # this is CLIP style
            out = out[torch.arange(out.shape[0]), input_array.argmax(dim=-1)]
        else:  # None
            pass

        if hasattr(self, "projection"):
            out = self.projection(out)

        return out, attentions


class Perceiver(nn.Module):
    def __init__(self, config):
        r"""Unassuming Perceiver Architecture that sits at the heart of this project. In practive this is a nice
        wrapper around model returned by ``get_sequential_from_config`` that automatically handles different
        types of input in a simple fashion. This is a great approach when using on a single GPU or performing
        Data Parallel training on multiple GPUs. When using this for Model Parallel training, you will need to
        write your own list etc. read story on `distributed <stories.2.html>`_ for more details.

        Args:
            config: ``gperc.PerceiverConfig`` object
        """

        super().__init__()
        self.config = config
        self.embd = Embeddings(config)
        self.encoder_block = EncoderBlock(config)
        self.processors = nn.ModuleList([ProcessorBlock(config) for _ in range(config.num_layers)])
        self.decoder_block = DecoderBlock(config)

    def num_parameters(self, include_non_trainable: bool = True):
        r"""function that returns the number of parameters in the modle

        Args:
            include_non_trainable (bool, optional): If true includes tensors that have ``requires_grad=False`` as well

        Returns:
            int: number of parameters in the model
        """
        if include_non_trainable:
            return sum(p.numel() for p in self.parameters())
        else:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, input_array, attention_mask=None, output_array=None, return_attentions=False):
        r"""Performs the forward pass of the Perceiver.

        Args:
            input_array (torch.Tensor): Input array to the Perceiver, read paper for reference
            attention_mask (torch.Tensor, optional): Mask for the decoder, attends at location with value 1
            output_array (torch.Tensor, optional): Output array to the Perceiver, read paper for reference
            return_attentions (bool, optional): If true returns the attentions as a list

        Returns:
            Tuple[torch.Tensor, List[torch.Tensor]] if ``return_attentions`` is True else torch.Tensor: \
                The output of the Perceiver and the attention matrices
        """

        def __check_conditionals():
            assert len(input_array.shape) in [2, 3], "Input array must be of shape [batch_size, input_len, (input_dim)]"

            # if isinstance(input_object, torch.tensor):
            #     return input_object, None, None
            # elif isinstance(input_object, tuple):
            #     if len(input_object) == 2:
            #         if isinstance(input_object[1], bool):
            #             return input_object[0], None, input_object[1]
            #         else:
            #             assert isinstance(input_object[1], torch.tensor), "The input_object must contain either "\
            #                 "(input_array, output_array) or (input_array, return_attentions)"
            #             return input_object[0], input_object[1], None
            #     else:
            #         assert len(input_object) == 3, "The input_object must contain either (input_array, output_array, return_attentions)"

        __check_conditionals()

        # step 1: pass through the embedding, there is no need to pass attention_mask here
        input_tensor, attention_mask, latent_array, output_array = self.embd(input_array, attention_mask, output_array)

        # step 2: pass through the encoder block
        latents, output_array, attentions = self.encoder_block(input_tensor, attention_mask, latent_array, output_array)

        # step 3: pass through the processor blocks
        for i, p_block in enumerate(self.processors):
            latents, _, attentions = p_block(latents, None, attentions)

        # step 4: pass through the decoder block
        logits, attentions = self.decoder_block(input_array, latents, output_array, attentions)

        # step 5: return the items
        if return_attentions:
            return logits, attentions
        return logits


def get_distributed_model(config):
    r"""This function returns the model that is used for distributed training. This is **not** a wrapper
    around ``Perceiver`` but instead returns a ``Pipe`` object.


    .. note::

        When using GPipe you need to send in tensors because it will try to send items as microbatches
        for each GPU. Now that requires all the inputs to be tensors, so here I have written some
        basic dumb heuristic that can set attention_mask and output_array to None if average of the values
        in those tensors is -69 and -420 resp.

        Image classification task does not require any ``attention_mask`` you can pass that as a tensor
        with values ``attention_mask = torch.tensor([-69. for _ in range(batch_size)])`` and similarly you
        can send ``output_array`` as a tensor with values ``output_array = torch.tensor([-420. for _ in range(batch_size)])``


    Args:
        config (PerceiverConfig): Configuration object for the Perceiver

    Returns:
        ``torch.distributed.pipeline.sync.Pipe``: Model that can be used inplace of ``Perceiver`` but note that
        it can only take in ``torch.Tensor`` objects and not ``None``.
    """
    import sys

    if sys.platform == "win32":
        print("Windows platform is not supported for pipeline parallelism")
        sys.exit(0)
    if torch.cuda.device_count() < 2:
        print("Need at least two GPU devices for this tutorial")
        sys.exit(0)

    from torch.distributed.pipeline.sync import Pipe

    num_gpus = torch.cuda.device_count()
    partition_len = ((config.num_layers - 1) // num_gpus) + 1

    # Add encoder in the beginning.
    tmp_list = [
        Embeddings(config).cuda(0),
        EncoderBlock(config).cuda(0),
    ]
    module_list = []

    # Add all the necessary transformer blocks.
    for i in range(config.num_layers):
        processor_block = ProcessorBlock(config)
        if i != 0 and i % (partition_len) == 0:
            # note how we need to provice nn.Sequential object to the Pipe object
            module_list.append(nn.Sequential(*tmp_list))
            tmp_list = []
        device = i // (partition_len)
        tmp_list.append(processor_block.to(device))

    # Add decoder in the end.
    tmp_list.append(DecoderBlock(config).cuda(num_gpus - 1))
    module_list.append(nn.Sequential(*tmp_list))

    # Build the pipeline that takes in a sequential of sequential object
    chunks = 1
    model = Pipe(nn.Sequential(*module_list), chunks=chunks)

    return model


# ====== use case specific models ====== #


class PerceiverMLM(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.emb = torch.nn.Embedding(config.vocab_size, config.input_dim)
        self.perc = Perceiver(config)

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        x = self.emb(x)
        logits = self.perc(x, x)
        return logits


class PerceiverImage(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.perceiver = Perceiver(config)

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        return self.perceiver(x)
