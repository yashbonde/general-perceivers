r"""
Perceiver Model
===============

This file has code on the neural network of the pervceiver architecture. ``gperc.models.Perceiver`` sits at
the heart of this project.
"""

import math
import torch
from torch import nn

from collections import OrderedDict


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

    def forward(self, kv, q):
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
        A = self.drop_att(A.softmax(dim=-1))  # [b, h, n, m]
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


class Perceiver(nn.Module):
    def __init__(self, config):
        r"""Unassuming Perceiver Architecture that sits at the heart of this project.

        Args:
            config: ``gperc.PerceiverConfig`` object
        """

        def __check_conditionals():
            assert config.decoder_cross_attention or config.decoder_projection, "Must have either cross attention or projection"
            if config.decoder_projection:
                assert hasattr(config, "n_classes") and config.n_classes, "Must have n_classes > 0 if using projection"

        __check_conditionals()

        super().__init__()
        self.config = config

        # define the 3 positional embeddings for the model input, latent, output
        self.pos_emb_encoder = build_position_encoding("trainable", config, config.input_len, config.input_dim)
        self.pos_emb_latent = build_position_encoding("trainable", config, config.latent_len, config.latent_dim)
        self.pos_emb_decoder = build_position_encoding("trainable", config, config.output_len, config.output_dim)

        # define the blocks
        self.encoder_block = Block(
            kv_dim=config.input_dim,
            q_dim=config.latent_dim,
            num_heads=config.num_heads,
            ffw_dim=config.ffw_latent,
            dropout=config.dropout,
            add_residual=True,
        )

        self.processors = nn.ModuleList(
            [
                Block(
                    kv_dim=config.latent_dim,
                    q_dim=config.latent_dim,
                    num_heads=config.num_heads,
                    ffw_dim=config.ffw_latent,
                    dropout=config.dropout,
                    add_residual=True,
                )
                for _ in range(config.num_layers)
            ]
        )

        if config.decoder_cross_attention:
            self.decoder_block = Block(
                kv_dim=config.latent_dim,
                q_dim=config.output_dim,
                num_heads=config.num_heads,
                ffw_dim=config.ffw_latent,
                dropout=config.dropout,
                add_residual=config.decoder_residual,
            )

        if config.decoder_projection:
            self.projection = nn.Linear(config.latent_dim, config.n_classes)

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

    def forward(self, input_array, output_array=None, return_attentions=None):
        r"""Performs the forward pass of the Perceiver.

        Args:
            input_array (torch.Tensor): Input array to the Perceiver, read paper for reference
            output_array (torch.Tensor, optional): Output array to the Perceiver, read paper for reference
            return_attentions (bool, optional): If true returns the attention matrices

        Returns:
            Tuple[torch.Tensor, List[torch.Tensor]] if ``return_attentions`` is True else torch.Tensor: \
                The output of the Perceiver and the attention matrices
        """

        def __check_conditionals():
            assert len(input_array.shape) == 3, "Input array must be of shape (batch_size, input_len, input_dim)"

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
        attentions = []

        # step 1: add positional encoding to the input_array
        input_array = input_array + self.pos_emb_encoder[None, : input_array.size(1), ...]

        # step 2: get the latent array i.e. nothing but the positional embedding concatenated
        latent_array = torch.cat([self.pos_emb_latent[None, ...] for _ in range(input_array.shape[0])], dim=0)

        # step 3: pass input and latent arrays through the encoder
        latents, enc_att = self.encoder_block(input_array, latent_array)
        attentions.append(enc_att)

        # step 4: pass the latents through the processor
        x = latents
        for i, p in enumerate(self.processors):
            x, A = p(x, x)
            attentions.append(A)

        # step 5: pass the latents and output_array through the decoder
        if output_array is None:
            # when user has not provided any output_array it is same as saying that just like latents the positional
            # embedding ends up becoming the output_array
            decoder_query = torch.cat([self.pos_emb_decoder[None, ...] for _ in range(latents.shape[0])], dim=0)
        else:
            # when the user has provided some output_array then add the positional aspect to it
            decoder_query = (
                output_array + self.pos_emb_decoder[None, : output_array.size(1), ...]
            )  # add the positional embedding to output array

        if self.config.decoder_cross_attention:
            out, A = self.decoder_block(latents, decoder_query)
            attentions.append(A)
        else:
            out = latents.mean(dim=1)

        if hasattr(self, "projection"):
            out = self.projection(out)

        if return_attentions:
            return out, attentions
        else:
            return out


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
