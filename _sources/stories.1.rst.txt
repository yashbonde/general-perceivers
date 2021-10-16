1. Quick Dopamine
=================

Perceiver can consume any kind of data that you give to this, we have added exmaples to test that hypothesis
out. I have added code for the following:

#. [`link <https://github.com/yashbonde/general-perceivers/blob/master/examples/train_cifar.py>`_] **Image classification:**
    Training a CIFAR10 model, the input image is flattened to a 2D array with shape ``[1024,3]`` and
    the latents are of shape ``[32,8]`` and finally classification happens on ``n_classes = 10`` after an
    average pooling across ``latent_len``. This is how config is set in ``gperc``

    .. code-block:: python

      from gperc import ImageConfig

      config = ImageConfig(
        image_shape=[32,32,3],
        latent_len=32,
        latent_dim=32,
        n_classes = 10,
      )

#. [`link <https://github.com/yashbonde/general-perceivers/blob/master/examples/train_lm.py>`_] **Masked Language Modeling:**
    Training a BERT model on a few articles, this happens using ``gperc``

    .. code-block:: python

      from gperc import TextConfig

      config = TextConfig(
        latent_dim = 8,
        vocab_size = len(vocabulary),
        max_len = tensor_target.shape[1],
        latent_frac = 0.15
      )

#. More 🍰 on the way
