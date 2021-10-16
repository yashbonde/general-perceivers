import torch
from torch.nn import functional as F

from gperc import Perceiver, PerceiverConfig
from gperc.utils import set_seed

import unittest

from tqdm import trange


class Test(unittest.TestCase):
    def test_cifar10_forward(self):
        config = PerceiverConfig(
            input_len=32 * 32,
            input_dim=3,
            latent_len=32,
            latent_dim=16,
            output_len=10,
            output_dim=32,
            decoder_len=1,
            decoder_projection=True,
            output_pos_enc=False,
            decoder_residual=False,
            n_classes=10,
        )
        model = Perceiver(config)

        out, attentions = model(torch.randn(1, config.input_len, config.input_dim), return_attentions=True)

        # check the shapes
        self.assertEquals(out.shape, (1, config.n_classes))
        for i, a in enumerate(attentions):
            if not i:
                self.assertEquals(a.shape, (1, config.num_heads, config.latent_len, config.input_len))
            else:
                self.assertEquals(a.shape, (1, config.num_heads, config.latent_len, config.latent_len))

    def test_image_overfit(self):
        config = PerceiverConfig(
            input_len=9 * 9,
            input_dim=3,
            latent_len=32,
            latent_dim=16,
            num_layers=1,
            output_len=2,
            output_dim=10,
            decoder_len=1,
            decoder_cross_attention=True,
            decoder_projection=False,
            output_pos_enc=False,
            decoder_residual=False,
            n_classes=10,
            seed=4,
        )
        print(config)
        set_seed(config.seed)
        model = Perceiver(config)

        optim = torch.optim.Adam(model.parameters(), lr=3e-4)

        x = torch.randn(3, config.input_len, config.input_dim)
        y = torch.randint(low=0, high=config.output_dim, size=(3,))

        pbar = trange(4000)
        all_loss = []
        for i in pbar:
            _y = model(x)[:, 0, :]
            # print(_y.shape)
            loss = F.cross_entropy(_y, y)
            all_loss.append(loss.item())
            pbar.set_description(f"loss: {loss.item():.4f} | max: {max(all_loss):.4f}")
            loss.backward()
            optim.step()

            if loss < 1e-5:
                break
