1. Quick Dopamine
=================

Perceiver can consume any kind of data that you give to this, the examples added here demonstrate how powerful
this approach can be. I have added code for the following:

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

#. [`link <https://github.com/yashbonde/general-perceivers/blob/master/examples/train_segnet.py>`_] **Image Segmentation:**
    Training a simple segmentation network on image segmentation `dataset  <https://www.robots.ox.ac.uk/~vgg/data/iseg/>`_.
    In ``gperc`` you need to define configuration as follows:

    .. code-block:: python

      from gperc import ImageConfig

      config = ImageConfig(
        image_shape=(224, 224, 3),
        latent_len=32,
        latent_dim=32,
        num_layers=6,
        n_classes=2,
        task="segmentation",
      )

#. [`link <https://github.com/yashbonde/general-perceivers/blob/master/examples/train_lm.py>`_] **Grokking:**
    Grokking is the phenomenon of model snapping into place and learn the rule as you keep on training,
    from `this <https://mathai-iclr.github.io/papers/papers/MATHAI_29_paper.pdf>`_ paper.

    .. code-block:: python

      from gperc import TextConfig

      config = TextConfig(
        latent_dim = 8,
        vocab_size = len(vocabulary),
        max_len = tensor_target.shape[1],
        latent_frac = 0.15
      )

#. [`link <https://github.com/yashbonde/general-perceivers/blob/master/examples/train_rl.py>`_] **Reinforcement Learning:**
    Train a simple model to solve Cart Pole gym environment.


#. [`link <https://github.com/yashbonde/general-perceivers/blob/master/examples/train_transfer.py>`_] **Transfer Learning: (WIP)**
    The real power of transformer encoder model comes from the fact that unsupervised training and transfering that to
    downstream task for classification helps build really powerful models. I want to have this functionality built into
    ``gperc`` directly. For this I have added a custom built from scratch ``gperc.Consumer`` object that manages your data.
    (WIP) finetuning also requires changing the architecture a little bit, this is also being added as first class in
    ``gperc.Perceiver`` object.

#. More 🍰 on the way
