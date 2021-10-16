"""
Data Reader
===========

This file has the datareaders for the ``gperc`` program. The datareaders work as follows:

#. The default datareader is ``gperc.BinaryConsumer`` and it reads all the files that are \
  provided to this. It reads the binaries of the files and reads the data based on pretty \
  much the size of the file.
#. You can provide extra metadata to the datareaders. This is done by providing a dictionary \
  or list. For more information read below.
"""
