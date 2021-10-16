from typing import List
from fire import Fire

import logging
from types import SimpleNamespace


class Main:
    def __init__(self, modality: str, seed: int = 4):
        """This is the main class for running any kind.

        **DO NOT USE (WIP)**

        Args:
            modality (str): what exactly is the purpose of this network
            seed (int, optional): seed for randomness. Defaults to 4.
        """
        self.modality = modality
        self.seed = seed

        raise NotImplementedError("CLI work in progress, please use the pacakge instead")

        # this is the keys
        self.__train_is_go = False
        self.__data_is_go = False
        self.__arch_is_go = False

    def train(self, batch_size: int, num_steps: int):
        """Config for training

        Args:
            batch_size (int): batch size
            num_steps (int): number of steps to train the model
        """
        self.train_config = SimpleNamespace(
            batch_size=batch_size,
            num_steps=num_steps,
        )
        self.__train_is_go = True

    def data(self, dataset_name: str, vocab_size: int = None, image_size: int = None):
        """Config for dataset

        Args:
            dataset_name (str): name of the dataset to load
            vocab_size (int, optional): size of the vocabulary, has to be provided if this has \
                ``text`` in modality.
            image_size (int, optional): size of the image dim, has to be provided if this has \
                ``image`` in the modality
        """
        self.data_config = SimpleNamespace(
            dataset_name=dataset_name,
            vocab_size=vocab_size,
            image_size=image_size,
        )
        self.__data_is_go = True

    def arch(
        self,
        mno: List,
        ced: List,
        ffw_width: float = 1.0,
    ):
        """Config for architecture. This is simple and takes in list of ints

        Args:
            mno (List): The first dimension of input, latent and output arrays
            ced (List): The second dimension of input, latent and output arrays
            ffw_width (float, optional): The width of the feed forward layer as ratio of dims
        """
        assert isinstance(mno, list), "mno must be a list"
        assert isinstance(ced, list), "ced must be a list"
        self.arch_config = SimpleNamespace(
            input_len=mno[0],
            input_dim=ced[0],
            latent_len=mno[1],
            latent_dim=ced[1],
            output_len=mno[2],
            output_dim=ced[2],
            ffw_latent=int(mno[1] * ffw_width),
            ffw_output=int(mno[2] * ffw_width),
        )
        self.__arch_is_go = True

    def start(self, log_level: int = 0):
        """This function triggers the CLI and shut has to be last in the chain

        Args:
            log_level (int, optional): log level in the order ``[logging.NOTSET, logging.DEBUG, \
                logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL]``
        """
        assert self.__train_is_go, "You need to add train config, do gperc train --help"
        assert self.__data_is_go, "You need to add data config, do gperc data --help"
        assert self.__arch_is_go, "You need to add arch config, do gperc arch --help"

        level = [
            logging.NOTSET,
            logging.DEBUG,
            logging.INFO,
            logging.WARNING,
            logging.ERROR,
            logging.CRITICAL,
        ][log_level]

        print("running this code")


if __name__ == "__main__":
    Fire(Main)
