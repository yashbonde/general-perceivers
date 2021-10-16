1. Quick Dopamine
=================

Perceiver can consume any kind of data that you give to this, we have added exmaples to test that hypothesis
out. I have added code for the following:


1. [`link <https://github.com/yashbonde/general-perceivers/blob/master/examples/train_cifar.py>`_]
**Image classification:**
Training a CIFAR10 model, the input image is flattened to a 2D array with shape ``[1024,3]`` and
the latents are of shape ``[32,64]``


1. [`link <https://github.com/yashbonde/general-perceivers/blob/master/examples/train_lm.py>`_]
**Masked Language Modeling:**
Training a BERT model on a few articles
    
