import random
import unittest
from tqdm import trange

import torch
from torch.nn import functional as F

from nbox.utils import folder, join

from gperc import Perceiver, PerceiverConfig

from gperc.configs import BinaryConfig, TextConfig, ImageConfig
from gperc.data import Consumer
from gperc.arrow import ArrowConsumer
from gperc.trainer import Trainer
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
            elif i == len(attentions) - 1:
                self.assertEqual(a.shape, (1, config.num_heads, config.output_len, config.latent_len))
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
            decoder_reduction="mean",
            decoder_projection=True,
            n_classes=10,
            decoder_residual=False,
            seed=4,
        )
        set_seed(config.seed)
        model = Perceiver(config)

        optim = torch.optim.SGD(model.parameters(), lr=3e-4, momentum=0.1)

        x = torch.randn(2, config.input_len, config.input_dim)
        y = torch.randint(low=0, high=config.output_dim, size=(2,))

        print("input shape:", x.shape)
        print("output shape:", y.shape)

        pbar = trange(4000)
        all_loss = []
        success = False
        for i in pbar:
            _y = model(x)
            loss = F.cross_entropy(_y, y)
            all_loss.append(loss.item())
            pbar.set_description(f"loss: {loss.item():.4f} | max: {max(all_loss):.4f} | min: {min(all_loss):.4f}")
            loss.backward()
            optim.step()

            if loss < 1e-5:
                success = True
                break

        self.assertTrue(success, msg="Failed image/classification overfit test. Something is wrong!")

    def test_mlm_forward(self):
        r"""It is very hasrd to debug if the attention mask is behaving properly or not, please
        check the code in gperc.Block for manual inspection."""
        config = TextConfig(latent_dim=8, latent_frac=0.5, vocab_size=100, max_len=6, num_heads=1)
        set_seed(4)
        lens = random.choices(range(config.max_len // 2, config.max_len), k=4)
        sequences = [random.choices(range(1, config.vocab_size), k=l) for l in lens]
        attention_masks = []
        for s in sequences:
            attention_masks.append([1] * len(s) + [0] * (config.max_len - len(s)))
            s.extend([0] * (config.max_len - len(s)))

        sequences = torch.tensor(sequences)
        attention_masks = torch.tensor(attention_masks)

        model = Perceiver(config)

        out, attentions = model(sequences, attention_masks, return_attentions=True)
        assert out.shape == (sequences.shape[0], config.max_len, config.vocab_size)
        for i, a in enumerate(attentions):
            if not i:
                assert a.shape == (sequences.shape[0], config.num_heads, config.latent_len, config.max_len)
            elif i == len(attentions) - 1:
                assert a.shape == (sequences.shape[0], config.num_heads, config.output_len, config.latent_len)
            else:
                assert a.shape == (sequences.shape[0], config.num_heads, config.latent_len, config.latent_len)

    def test_mlm_overfit(self):
        r"""It is very hasrd to debug if the attention mask is behaving properly or not, please
        check the code in gperc.Block for manual inspection."""
        config = TextConfig(
            latent_dim=8,
            latent_frac=0.5,
            vocab_size=32,
            max_len=10,
        )
        set_seed(4)
        lens = random.choices(range(config.max_len // 2, config.max_len), k=4)
        sequences = [random.choices(range(1, config.vocab_size), k=l) for l in lens]
        attention_masks = []
        for s in sequences:
            attention_masks.append([1] * len(s) + [0] * (config.max_len - len(s)))
            s.extend([0] * (config.max_len - len(s)))

        sequences = torch.tensor(sequences)
        attention_masks = torch.tensor(attention_masks)

        model = Perceiver(config)

        optim = torch.optim.Adam(model.parameters(), lr=3e-4)

        target = sequences.clone().reshape(-1)
        sequences[torch.randn(*sequences.shape) < 0.15] = 0

        print("input shape:", sequences.shape)
        print("attention mask shape:", attention_masks.shape)

        pbar = trange(10_000)
        all_loss = []
        success = False
        for i in pbar:
            optim.zero_grad()
            _y = model(sequences).reshape(-1, config.vocab_size)
            loss = F.cross_entropy(_y, target.reshape(-1))
            all_loss.append(loss.item())
            pbar.set_description(f"loss: {loss.item():.4f} | max: {max(all_loss):.4f} | min: {min(all_loss):.4f}")
            loss.backward()
            optim.step()

            if loss < 0.01:
                success = True
                break

        self.assertTrue(success, msg="Failed text/mlm overfit test. Something is wrong!")


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
        print(data)

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
        data = Consumer(dataset, seqlen=128, style="concat", verbose=True, class_to_id={"tinker": 0, "tailor": 1, "soldier": 2, "spy": 3})

        # check the shapes for cases I0-I5
        out = data[None]
        self.assertTrue(set(out.keys()) == {"input_array", "attention_mask"})
        self.assertEqual(out["input_array"].shape, (1, 128))  # I0

        out = data[random.choice(range(len(data)))]
        self.assertTrue(set(out.keys()) == {"input_array", "attention_mask"})
        self.assertEqual(out["input_array"].shape, (1, 128))  # I1

        out = data[4:5]
        self.assertTrue(set(out.keys()) == {"input_array", "attention_mask"})
        self.assertEqual(out["input_array"].shape, (1, 128))  # I2 - A

        out = data[0:5]
        self.assertTrue(set(out.keys()) == {"input_array", "attention_mask"})
        self.assertEqual(out["input_array"].shape, (5, 128))  # I2 - B

        out = data[[1, 2, 3, 4, 5]]
        self.assertTrue(set(out.keys()) == {"input_array", "attention_mask"})
        self.assertEqual(out["input_array"].shape, (5, 128))  # I3

        out = data[{"tinker": 1, "tailor": 2, "soldier": 1, "spy": 1}]
        self.assertTrue(set(out.keys()) == {"input_array", "attention_mask"})
        self.assertEqual(out["input_array"].shape, (5, 128))  # I4

        out = data[{"tinker": [1, 2], "tailor": [0, 1], "soldier": [0], "spy": [1]}]
        self.assertTrue(set(out.keys()) == {"input_array", "attention_mask"})
        self.assertEqual(out["input_array"].shape, (6, 128))  # I5

        # check the shapes for case I6 -> supervised
        data.set_supervised_mode()
        out = data[{"tinker": 1, "tailor": 2, "soldier": 1, "spy": 1}, "supervised"]
        self.assertTrue(set(out.keys()) == {"input_array", "attention_mask", "class"})
        self.assertEqual(out["input_array"].shape, (5, 128))  # I4
        self.assertEqual(out["class"].shape, (5,))  # I4

        data.set_unsupervised_mode()
        out = data[{"tinker": [1, 2], "tailor": [0, 1], "soldier": [0], "spy": [1]}, "unsupervised"]
        self.assertTrue(set(out.keys()) == {"input_array", "attention_mask", "output_array"})
        self.assertEqual(out["input_array"].shape, (6, 128))  # I5
        self.assertEqual(out["output_array"].shape, (6, 128))  # I5


class TestArrowDataset(unittest.TestCase):
    def test_dataset_full(self):
        # this method tests the Consumer on itself, going to docs
        folder_path = join(folder(__file__), "docs", "source")
        labels = ["tinker", "tailor", "soldier", "spy"]  # some labels
        dataset = {}
        for i, f in enumerate(get_files_in_folder(folder_path, [".rst"], sort=True)):
            dataset[f] = labels[i % len(labels)]

        # create the dataset
        data = ArrowConsumer(dataset, seqlen=128, class_to_id={"tinker": 0, "tailor": 1, "soldier": 2, "spy": 3})

        # check the shapes for cases I0-I5
        out = data[None]
        self.assertTrue(set(out.keys()) == {"input_array", "attention_mask", "class"})
        self.assertEqual(out["input_array"].shape, (1, 128))  # I0

        out = data[random.choice(range(len(data)))]
        self.assertTrue(set(out.keys()) == {"input_array", "attention_mask", "class"})
        self.assertEqual(out["input_array"].shape, (1, 128))  # I1

        out = data[4:5]
        self.assertTrue(set(out.keys()) == {"input_array", "attention_mask", "class"})
        self.assertEqual(out["input_array"].shape, (1, 128))  # I2 - A

        out = data[0:5]
        self.assertTrue(set(out.keys()) == {"input_array", "attention_mask", "class"})
        self.assertEqual(out["input_array"].shape, (5, 128))  # I2 - B

        out = data[[1, 2, 3, 4, 5]]
        self.assertTrue(set(out.keys()) == {"input_array", "attention_mask", "class"})
        self.assertEqual(out["input_array"].shape, (5, 128))  # I3

        # out = data[{"tinker": 1, "tailor": 2, "soldier": 1, "spy": 1}]
        # self.assertTrue(set(out.keys()) == {"input_array", "attention_mask", "class"})
        # self.assertEqual(out["input_array"].shape, (5, 128))  # I4

        # out = data[{"tinker": [1, 2], "tailor": [0, 1], "soldier": [0], "spy": [1]}]
        # self.assertTrue(set(out.keys()) == {"input_array", "attention_mask", "class"})
        # self.assertEqual(out["input_array"].shape, (6, 128))  # I5

        # # check the shapes for case I6 -> supervised
        # data.set_supervised_mode()
        # out = data[{"tinker": 1, "tailor": 2, "soldier": 1, "spy": 1}, "supervised"]
        # self.assertTrue(set(out.keys()) == {"input_array", "attention_mask", "class"})
        # self.assertEqual(out["input_array"].shape, (5, 128))  # I4
        # print(out["class"].shape)
        # self.assertEqual(out["class"].shape, (5,))  # I4

        # data.set_unsupervised_mode()
        # out = data[{"tinker": [1, 2], "tailor": [0, 1], "soldier": [0], "spy": [1]}, "unsupervised"]
        # self.assertTrue(set(out.keys()) == {"input_array", "attention_mask", "output_array"})
        # self.assertEqual(out["input_array"].shape, (6, 128))  # I5
        # print(out["output_array"].shape)
        # self.assertEqual(out["output_array"].shape, (6, 128))  # I5


class TestTrainer(unittest.TestCase):
    def test_trainer(self):
        folder_path = join(folder(__file__), "docs", "source")
        labels = ["tinker", "tailor", "soldier", "spy"]  # some labels
        class_to_id={"tinker": 0, "tailor": 1, "soldier": 2, "spy": 3}
        label_to_file = {}
        for i, f in enumerate(get_files_in_folder(folder_path, [".rst"], sort=True)):
            label_to_file[labels[i % len(labels)]] = f
            if i == 4:
                break
        dataset = {v:k for k, v in label_to_file.items()}

        # create the dataset
        data = ArrowConsumer(dataset, seqlen = 128, n_bytes=1, class_to_id=class_to_id)
        print(data)

        # create config
        config = BinaryConfig(
            seqlen = data.seqlen,
            vocab_size = data.vocab_size,
            latent_dim = 16,
            latent_frac = 0.1,
            n_classes = len(class_to_id),
            ffw_ratio = 2.0,
            num_heads = 2,
            num_layers = 6,
            decoder_reduction = "mean"
        )
        print(config)
        set_seed(config.seed)

        # create model
        model = Perceiver(config)
        print(model.num_parameters())

        # create trainer
        trainer = Trainer(model, None)
        data.create_batches(batch_size=32)
        pbar = trange(10000)
        acc_over_time = []
        for i in pbar:
            meta = trainer(
                batch = data.get_next_batch(),
                step = 1,
                n_bytes = data.n_bytes,
                n_classes = data.n_classes,
                pbar = pbar,
                grad_clip = 1.0,
                optim = torch.optim.Adam(
                    model.parameters(),
                    lr=0.0001
                )
            )

            acc_over_time.append(meta["train/acc_avg"])

            if sum(acc_over_time[-10:]) == 10:
                break
