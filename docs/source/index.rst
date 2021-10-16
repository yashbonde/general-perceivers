.. gperc documentation master file, created by
   sphinx-quickstart on Sat Oct 16 god knows when, I am copy pasting this!
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to gperc's documentation!
=================================

``gperc`` simplifies using `PerceiverIO <https://deepmind.com/research/open-source/perceiver-IO>`_ 
an architecture by DeepMind which does shape transformation as ``mno,cde`` ie. it comsumes a shape 
``m,c``, converts it to latents ``n,d`` and finally transforms it to ``o,e``. All transformations happen
using self-attention mechanism in an encoder (``gperc.Encoder``) → processor (``gperc.Processor``) → 
decoder (``gperc.Decoder``) format. This allows it to get away with very long sequences that cannot
usually be managed by `vanilla transformers <https://arxiv.org/pdf/1706.03762.pdf>`_.

The simplicity of its formula along with general improvement of field means it is a higly practical tool
given sufficient data.

.. image:: assets/structure.png
   :alt: Alternative text

Indices and tables
==================

.. toctree::
   :maxdepth: 2
   :caption: Stories

   stories


.. toctree::
   :maxdepth: 2
   :caption: Documentation

   gperc.configs
   gperc.data
   gperc.models
   gperc.utils
   


* :ref:`genindex`
* :ref:`modindex`
