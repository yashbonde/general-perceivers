import random
import unittest
from tqdm import trange

import torch
from torch.nn import functional as F

from nbox.utils import folder, join

from gperc import Perceiver, PerceiverConfig

from gperc.configs import TextConfig, ImageConfig
from gperc.data import Consumer
from gperc.utils import set_seed, get_files_in_folder


class TestModel(unittest.TestCase):
    def test_cifar10_forward(self):
        config = ImageConfig(
            image_shape=(32, 32, 3),
            latent_len=16,
            latent_dim=4,
            n_classes=10,
            task="classification",
        )
        set_seed(4)
        model = Perceiver(config)

        out, attentions = model(torch.randn(1, config.input_len, config.input_dim), return_attentions=True)

        # check the shapes
        self.assertEqual(out.shape, (1, config.n_classes))
        for i, a in enumerate(attentions):
            if not i:
                self.assertEqual(a.shape, (1, config.num_heads, config.latent_len, config.input_len))
            else:
                self.assertEqual(a.shape, (1, config.num_heads, config.latent_len, config.latent_len))

    def test_image_overfit(self):
        config = PerceiverConfig(
            input_len=9 * 9,
            input_dim=3,
            latent_len=16,
            latent_dim=4,
            num_layers=2,
            output_len=1,
            output_dim=4,
            decoder_len=1,
            decoder_cross_attention=True,
            decoder_projection=False,
            output_pos_enc=False,
            decoder_residual=False,
            seed=4,
        )
        set_seed(config.seed)
        model = Perceiver(config)

        optim = torch.optim.Adam(model.parameters(), lr=3e-4)

        x = torch.randn(2, config.input_len, config.input_dim)
        y = torch.randint(low=0, high=config.output_dim, size=(2,))

        print("input shape:", x.shape)
        print("output shape:", y.shape)

        pbar = trange(4000)
        all_loss = []
        success = False
        for i in pbar:
            _y = model(x)[:, 0, :]
            loss = F.cross_entropy(_y, y)
            all_loss.append(loss.item())
            pbar.set_description(f"loss: {loss.item():.4f} | max: {max(all_loss):.4f}")
            loss.backward()
            optim.step()

            if loss < 1e-5:
                success = True
                break

        self.assertTrue(success, msg="Failed overfit test. Something is wrong!")


class TestDataset(unittest.TestCase):
    # read more about data here: https://yashbonde.github.io/general-perceivers/gperc.data.html
    def test_dataset_f0_mode(self):
        data = [f"file{i}" for i in range(10)]
        data = Consumer(data, _unittesting=True)
        assert data._mode == "F0"

    def test_dataset_f1_mode(self):
        data = [
            {"file0_0": "cat0"},
            {"file1_0": "cat0"},
            {"file2_0": "cat0"},
            {"file3_0": "cat0"},
            {"file0_1": "cat1"},
            {"file1_1": "cat1"},
            {"file2_1": "cat1"},
            {"file3_1": "cat1"},
        ]
        data = Consumer(data, _unittesting=True)
        assert data._mode == "F1"

    def test_dataset_f2_mode(self):
        data = {"file0": "cat3", "file1": "cat3", "file2": "cat2", "file3": "cat3", "file4": "cat1", "file5": "cat0"}
        data = Consumer(data, _unittesting=True)
        assert data._mode == "F2"

    def test_dataset_f3_mode(self):
        data = {f"cat{j}": [f"file{i}" for i in range(4)] for j in range(3)}
        data = Consumer(data, _unittesting=True)
        assert data._mode == "F3"

    @unittest.expectedFailure
    def test_dataset_fail_mode_0(self):
        # this will fail
        data = Consumer({"cat1": ["file1.txt", "file2.txt"], "cat2": "file3.txt"})

    # ----- usage ----- #

    def test_dataset_f0_getitem(self):
        data = [f"file{i}" for i in range(10)]
        data = Consumer(data, _unittesting=True)

        out = data[None]  # "file1.txt"                  # I0
        out = data[1]  # "file2.txt"                     # I1
        out = data[:1]  # ["file1.txt"]                  # I2
        out = data[[0, 1]]  # ["file1.txt", "file2.txt"] # I3
        with self.assertRaises(ValueError):
            out = data[{"cat1": 90}]  # I4
        with self.assertRaises(ValueError):
            out = data[{"cat1": [0, 1]}]  # I5

    def test_dataset_f1_getitem(self):
        data = [
            {"file0_0": "cat0"},
            {"file1_0": "cat0"},
            {"file2_0": "cat0"},
            {"file3_0": "cat0"},
            {"file0_1": "cat1"},
            {"file1_1": "cat1"},
            {"file2_1": "cat1"},
            {"file3_1": "cat1"},
        ]
        data = Consumer(data, _unittesting=True)

        out = data[None]  # "file1.txt"                  # I0
        out = data[1]  # "file2.txt"                     # I1
        out = data[:1]  # ["file1.txt"]                  # I2
        out = data[[0, 1]]  # ["file1.txt", "file2.txt"] # I3
        out = data[{"cat1": 3}]  # I4
        out = data[{"cat1": [0, 1]}]  # I5

    def test_dataset_f2_getitem(self):
        data = {"file0": "cat3", "file1": "cat3", "file2": "cat2", "file3": "cat3", "file4": "cat1", "file5": "cat0"}
        data = Consumer(data, _unittesting=True)

        out = data[None]  # "file1.txt"                  # I0
        out = data[1]  # "file2.txt"                     # I1
        out = data[:1]  # ["file1.txt"]                  # I2
        out = data[[0, 1]]  # ["file1.txt", "file2.txt"] # I3
        out = data[{"cat3": 2}]  # I4
        out = data[{"cat3": [0, 1]}]  # I5

    def test_dataset_f3_getitem(self):
        data = {f"cat{j}": [f"file{i}" for i in range(4)] for j in range(3)}
        data = Consumer(data, _unittesting=True)

        out = data[None]  # "file1.txt"                  # I0
        out = data[1]  # "file2.txt"                     # I1
        out = data[:1]  # ["file1.txt"]                  # I2
        out = data[[0, 1]]  # ["file1.txt", "file2.txt"] # I3
        out = data[{"cat1": 2}]  # I4
        out = data[{"cat2": [0, 1]}]  # I5

    def test_dataset_full(self):
        # this method tests the Consumer on itself, going to docs
        folder_path = join(folder(__file__), "docs", "source")
        labels = ["tinker", "tailor", "soldier", "spy"]  # some labels
        dataset = {}
        for i, f in enumerate(get_files_in_folder(folder_path, [".rst"], sort=True)):
            dataset[f] = labels[i % len(labels)]

        # create the dataset
        data = Consumer(dataset, seqlen=128, verbose=True, class_to_id={"tinker": 0, "tailor": 1, "soldier": 2, "spy": 3})

        # check the shapes for cases I0-I5
        out = data[None]
        self.assertTrue(set(out.keys()) == {"input_array"})
        self.assertEqual(out["input_array"].shape, (1, 128))  # I0

        out = data[random.choice(range(len(data)))]
        self.assertTrue(set(out.keys()) == {"input_array"})
        self.assertEqual(out["input_array"].shape, (1, 128))  # I1

        out = data[4:5]
        self.assertTrue(set(out.keys()) == {"input_array"})
        self.assertEqual(out["input_array"].shape, (1, 128))  # I2 - A

        out = data[0:5]
        self.assertTrue(set(out.keys()) == {"input_array"})
        self.assertEqual(out["input_array"].shape, (5, 128))  # I2 - B

        out = data[[1, 2, 3, 4, 5]]
        self.assertTrue(set(out.keys()) == {"input_array"})
        self.assertEqual(out["input_array"].shape, (5, 128))  # I3

        out = data[{"tinker": 1, "tailor": 2, "soldier": 1, "spy": 1}]
        self.assertTrue(set(out.keys()) == {"input_array"})
        self.assertEqual(out["input_array"].shape, (5, 128))  # I4

        out = data[{"tinker": [1, 2], "tailor": [0, 1], "soldier": [0], "spy": [1]}]
        self.assertTrue(set(out.keys()) == {"input_array"})
        self.assertEqual(out["input_array"].shape, (6, 128))  # I5

        # check the shapes for case I6 -> supervised
        data.set_supervised_mode()
        out = data[{"tinker": 1, "tailor": 2, "soldier": 1, "spy": 1}, "supervised"]
        self.assertTrue(set(out.keys()) == {"input_array", "class"})
        self.assertEqual(out["input_array"].shape, (5, 128))  # I4
        self.assertEqual(out["class"].shape, (5,))  # I4

        data.set_unsupervised_mode()
        out = data[{"tinker": [1, 2], "tailor": [0, 1], "soldier": [0], "spy": [1]}, "unsupervised"]
        self.assertTrue(set(out.keys()) == {"input_array", "output_array"})
        self.assertEqual(out["input_array"].shape, (6, 128))  # I5
        self.assertEqual(out["output_array"].shape, (6, 128))  # I5
