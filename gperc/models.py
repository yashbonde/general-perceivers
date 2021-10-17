"""
Perceiver Model
===============

This file has code on the neural network of the pervceiver architecture. ``gperc.models.Perceiver`` sits at \
    the heart of this operation.
"""


import torch
from torch import nn
from typing import Callable, List, Tuple


def build_position_encoding(position_encoding_type: str, config: dict, num_index_items: int, emb_dim: int):
    r"""Get the positional encoding matrix. If ``position_encoding_type == "trainable"`` then a random normal
    matrix is returned, if it is "sinusoid" then

    Args:
        position_encoding_type (str): [description]
        config: Config
        num_index_items (int): number of items in the embedding, eg. ``vocab_size``
        emb_dim (int): embedding dimension

    Returns:
        ``torch.nn.Parameter``: Item that can be used as a parameter in a ``torch.nn.Embedding``
    """
    if position_encoding_type == "trainable":
        # first define the positional encodings
        latent_pos_emb = nn.Parameter(torch.normal(mean=0.0, std=config.pos_init_std, size=(num_index_items, emb_dim)))
        return latent_pos_emb
    elif position_encoding_type == "sinusoid":
        # then define the positional encodings
        def get_pos_encoding(position):
            return torch.sin(position / (10000 ** (2 * (position / num_index_items))))

        return nn.Parameter(torch.stack([get_pos_encoding(i) for i in range(num_index_items)]).unsqueeze(0))


class Block(nn.Module):
    def __init__(self, kv_dim: int, q_dim: int, num_heads: int, ffw_dim: int, dropout: float = 0.0, add_residual: bool = False):
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

        # print("<<<<<<<", self.dim, self.q_dim)

    def forward(self, kv, q) -> Tuple[torch.Tensor, torch.Tensor]:
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
        # print("kv", kv.shape)
        # print("q", q.shape)

        _q = self.lnq(q)
        _kv = self.lnkv(kv)

        # print("q:", q.shape)
        # print("kv:", kv.shape)

        # then compute the query, key, value and split for multihead attention
        Q, K, V = self.fq(_q), self.fk(_kv), self.fv(_kv)
        # Q = einops(Q, 'b n d -> b h n m', d = self.q_dim, n = self.num_heads, m = self.q_dim // self.num_heads)
        # K = einops.rearrange(K, 'b n d -> b h n m', d = self.kv_dim, n = self.num_heads, m = self.kv_dim // self.num_heads)
        # V = einops.rearrange(V, 'b n d -> b h n m', d = self.kv_dim, n = self.num_heads, m = self.kv_dim // self.num_heads)
        Q, K, V = tuple(map(lambda x: x.view(x.shape[0], self.num_heads, -1, x.shape[-1] // self.num_heads), (Q, K, V)))
        # print("Q:", Q.shape)
        # print("K:", K.shape)
        # print("V:", V.shape)
        # print(K.permute(0, 1, 3, 2).shape)
        A = Q @ K.permute(0, 1, 3, 2) * (self.dim ** -0.5)  # [b, h, n, e/h] @ [b, h, e/h, m] -> [b, h, n, m]
        A = self.drop_att(A.softmax(dim=-1))  # [b, h, n, m]
        # print("A:", A.shape)
        # print((A @ V).shape)
        out = (A @ V).reshape((q.shape[0], -1, self.q_dim))  # [b, h, n, m] @ [b, h, m, e/h] -> [b, h, n, e/h] -> [b, n, e]
        # print("out:", out.shape)
        out = self.fo(out)
        # print("out:", out.shape)

        # Optionally include a residual to the query.
        # Consider omitting the residual if the semantics of query and output
        # are different, e.g. if queries are positions and outputs are pixels.
        if self.add_residual:
            out = out + q

        # print(">>>>>>>>", out.shape)

        # now we will pass it through the mlp
        out = self.mlp(self.lnqkv(out)) + out
        out = self.drop_mlp(out)

        return out, A


class Encoder(nn.Module):
    def __init__(self, config):
        r"""Generic Encoder Block of the model which takes in ``input_array`` as key-value and \
            ``latent_array`` as query.

        Args:
            config: Config
        """
        super().__init__()
        self.encoder_block = Block(
            kv_dim=config.input_dim,
            q_dim=config.latent_dim,
            num_heads=config.num_heads,
            ffw_dim=config.ffw_latent,
            dropout=config.dropout,
            add_residual=True,
        )

    def forward(self, input_array, latent_query) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        r"""Performs the forward pass of the encoder block.

        Args:
            input_array (torch.Tensor): Input array to the Encoder block, read paper for reference
            latent_query (torch.Tensor): Latent query to the Encoder block, read paper for reference

        Returns:
            Tuple[torch.Tensor, List[torch.Tensor]]: The output of the encoder block and the attention matrices
        """
        out, A = self.encoder_block(input_array, latent_query)
        return out, [A]


class Processor(nn.Module):
    def __init__(self, config):
        r"""Generic Processor Block of the model which takes in ``latent_array`` as key-value-query

        Args:
            config: Config
        """
        super().__init__()
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

    def forward(self, x) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        r"""Performs the forward pass of the processor block.

        Args:
            x (torch.Tensor): Input array to the Processor block, this should always be the latent_array

        Returns:
            Tuple[torch.Tensor, List[torch.Tensor]]: The output of the processor block and the attention matrices
        """
        attentions = []
        for i, processor in enumerate(self.processors):
            x, A = processor(x, x)
            attentions.append(A)
        return x, attentions


class Decoder(nn.Module):
    def __init__(self, config):
        r"""Generic Decoder Block of the model which takes in ``latent_array`` as key-value and \
            ``output_array`` as query.

        Args:
            config: Config
        """
        super().__init__()

        def __check_conditionals():
            assert config.decoder_cross_attention or config.decoder_projection, "Must have either cross attention or projection"
            if config.decoder_projection:
                assert hasattr(config, "n_classes") and config.n_classes, "Must have n_classes > 0 if using projection"

        __check_conditionals()
        self.config = config

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

    def forward(self, latents, decoder_query) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        r"""Performs the forward pass of the decoder block.

        Args:
            latents (torch.Tensor): Latent array to the Decoder block, read paper for reference
            decoder_query (torch.Tensor): Output array to the Decoder block, read paper for reference

        Returns:
            Tuple[torch.Tensor, List[torch.Tensor]]: The output of the decoder block and the attention matrices
        """
        attentions = []
        if self.config.decoder_cross_attention:
            x, A = self.decoder_block(latents, decoder_query)
            attentions.append(A)
        else:
            x = latents.mean(dim=1)

        if hasattr(self, "projection"):
            x = self.projection(x)

        return x, attentions


class Perceiver(nn.Module):
    def __init__(self, config, input_preprocessing: Callable = None, output_postprocessing: Callable = None):
        r"""Unassuming Perceiver Architecture that sits at the heart of this project.

        Args:
            config: Config
            input_preprocessing (Callable, optional): callable object that takes in ``input_array`` and performs \
                operation on it.
            output_postprocessing (Callable, optional): callable object that takes in ``output_array`` and performs \
                operation on it.
        """
        super().__init__()
        self.config = config
        self.input_preprocessing = input_preprocessing
        self.output_postprocessing = output_postprocessing

        self.pos_emb_latent = build_position_encoding("trainable", config, config.latent_len, config.latent_dim)
        self.pos_emb_decoder = build_position_encoding("trainable", config, config.output_len, config.output_dim)

        self.encoder = Encoder(config)
        self.processor = Processor(config)
        self.decoder = Decoder(config)

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

    def forward(self, input_array: torch.Tensor, output_array: torch.Tensor = None, return_attentions: bool = False):
        r"""Performs the forward pass of the Perceiver.

        Args:
            input_array (torch.Tensor): Input array to the Perceiver, read paper for reference
            output_array (torch.Tensor, optional): Output array to the Perceiver, read paper for reference
            return_attentions (bool, optional): If true returns the attention matrices

        Returns:
            Tuple[torch.Tensor, List[torch.Tensor]] if ``return_attentions`` is True else torch.Tensor: \
                The output of the Perceiver and the attention matrices
        """
        if self.input_preprocessing:
            input_array = self.input_preprocessing(input_array)

        assert len(input_array.shape) == 3, "Input array must be of shape (batch_size, input_len, input_dim)"

        # enc -> proc -> decode
        latent_array = torch.cat([self.pos_emb_latent[None, ...] for _ in range(input_array.shape[0])], dim=0)
        latents, enc_att = self.encoder(input_array, latent_array)
        latents, proc_att = self.processor(latents)

        if output_array is None:
            decoder_query = torch.cat([self.pos_emb_decoder[None, ...] for _ in range(latents.shape[0])], dim=0)
        else:
            decoder_query = output_array + self.pos_emb_decoder[None, ...]  # add the positional embedding to output array
        out, dec_att = self.decoder(latents, decoder_query)

        if self.output_postprocessing:
            out = self.output_postprocessing(out)
        if return_attentions:
            return out, [*enc_att, *proc_att, *dec_att]
        else:
            return out
