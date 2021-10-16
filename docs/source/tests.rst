Tests
=====

Neural networks are tricky to train and with something like ``gperc`` where the entirity of code is just
writing big config files (**not YAML** 😛), testing of underlying layer is very crucial. To that extent
I have written tests for:

#. ``test_cifar10_forward``: to test if it can process CIFAR10 dataset
#. ``test_image_overfit``: to test if we overfit the model on a small random dataset, can it's loss go all
the way to zero.
