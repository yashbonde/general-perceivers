from pprint import pformat, pprint as pp


class PerceiverConfig:
    def __init__(self, **kwargs):
        # since perciever is such a powerful and versatile model, we need a good
        # config for this. I guess for different application we will simply define
        # different configurations and wrap them in some model registry-kinda thing

        # is there some pre-processing to be done?
        self.pre_processing = None

        # [MC => ND => OE]

        self.input_len = 64  # M: number of inputs
        self.input_dim = 8  # C: input embedding size

        self.pos_init_std = 0.02  # Standard normal initialization for the position encoding
        self.num_heads = 2  # H: number of heads in the attention blocks
        self.num_layers = 2  # L: number of process layers
        self.latent_len = 4  # N: number of latents
        self.latent_dim = 16  # D: latent size
        self.ffw_latent = 32  # FFW hidden dimension for latents

        self.dropout = 0.1 # dropout rate

        self.decoder_cross_attention = False  # is decoder a
        self.output_len = 1  # O: number of outputs during pretraining
        self.output_dim = 10  # E: dimension of latent queries
        self.decoder_residual = False  # whether to add the output query to the decoder
        self.output_pos_enc = False  # whether to add position encoding for the output
        self.decoder_projection = True  # is there a projection layer after the decoder

        self.seed = 4 # random seed

        # is there some post processing to be done?
        self.post_processing = None

        for k, v in kwargs.items():
            setattr(self, k, v)

    def __repr__(self) -> str:
        return pformat(self.__dict__, indent=2, sort_dicts=True)


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
