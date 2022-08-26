# We have to compare the serialisation and deserialisation of numpy arrays

import os
from timeit import timeit
from tempfile import gettempdir
import numpy as np
from data_pb2 import NDArray
from common import array_to_buffer, buffer_to_array

TRIES = 100

# random matrix functions
randn_float = lambda x, n: np.random.randn(*(x for _ in range(n))).astype(np.float32)
randn_string = lambda x, n: np.array([
  "".join(map(chr, np.random.randint(0, 256, size=x))) for _ in range(n)
])

# create the folders for writing files in
folder = gettempdir() + "/protobuf_numpy_test_serialisation"
protofolder = folder + "/protobufs"
arrayfolder = folder + "/arrays"
os.makedirs(protofolder, exist_ok=True)
os.makedirs(arrayfolder, exist_ok=True)


def array_to_proto(x, fp):
  np_proto = array_to_buffer(x)
  with open(fp, "wb") as f:
    f.write(np_proto.SerializeToString())

def proto_to_array(fp):
  np_proto = NDArray()
  with open(fp, "rb") as f:
    np_proto.ParseFromString(f.read())
  buffer_to_array(np_proto)

def array_to_npz(x, fp):
  np.savez(fp, x)

def npz_to_array(fp):
  np.load(fp)["arr_0"]

# now start timing

_array_to_proto = []
_proto_to_array = []
_array_to_npz = []
_npz_to_array = []
_x_items = [
  10, 12, 15, 20, 25, 50, 75, 100, 150, 200, 300, 400,  500, 600, 700, 800, 900, 1000,
  1010, 1020, 1030, 1040, 1050, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 3000
]

fn_ = [("float32", randn_float), ("string", randn_string)]

fn_ = fn_[:1] # string works weirdly with numpy, so we are skipping the serialisation tests

for title, _fn_ in fn_:
  for x in _x_items:
    # 7d because 7 elements is a million elements
    _arr = _fn_(x, 2)

    print(f"2D | array_to_proto | {x:04d} | {x*x:07d} | ", end = "")
    out = timeit(
      "array_to_proto(_arr, '{}/{}.proto')".format(protofolder, x),
      setup="from __main__ import array_to_proto, _arr",
      number=TRIES
    )
    file_size = os.path.getsize(protofolder + "/" + str(x) + ".proto")
    file_size = file_size / 1024
    print(f"{out:.4f}s | {file_size:04.2f}KB")
    _array_to_proto.append(out)

    print(f"2D | proto_to_array | {x:04d} | {x*x:07d} | ", end = "")
    out = timeit(
      "proto_to_array('{}/{}.proto')".format(protofolder, x),
      setup="from __main__ import proto_to_array",
      number=TRIES
    )
    print(f"{out:.4f}s")
    _proto_to_array.append(out)

    print(f"2D |   array_to_npz | {x:04d} | {x*x:07d} | ", end = "")
    out = timeit(
      "array_to_npz(_arr, '{}/{}.npz')".format(arrayfolder, x),
      setup="from __main__ import array_to_npz, _arr",
      number=TRIES
    )
    file_size = os.path.getsize(arrayfolder + "/" + str(x) + ".npz")
    file_size = file_size / 1024
    print(f"{out:.4f}s | {file_size:04.2f}KB")
    _array_to_npz.append(out)

    print(f"2D |   npz_to_array | {x:04d} | {x*x:07d} | ", end = "")
    out = timeit(
      "npz_to_array('{}/{}.npz')".format(arrayfolder, x),
      setup="from __main__ import npz_to_array",
      number=TRIES
    )
    print(f"{out:.4f}s")
    _npz_to_array.append(out)

  # remove all the files
  print("Removing all the files")
  for f in os.listdir(protofolder):
    os.remove(protofolder + "/" + f)

  # create a chart in matplotlib

  import matplotlib.pyplot as plt

  _x_items_ = [x*x for x in _x_items]

  plt.plot(_x_items_, _array_to_proto, color = "blue", label="array_to_proto")
  plt.plot(_x_items_, _proto_to_array, "--", color = "blue", label="proto_to_array")
  plt.plot(_x_items_, _array_to_npz, color = "orange", label="array_to_npz")
  plt.plot(_x_items_, _npz_to_array, "--", color = "orange", label="npz_to_array")
  plt.legend()

  plt.xlabel("Number of elements in matrix")
  plt.ylabel("Time (s)")
  plt.title(f"Serialisation [{title}] protobuf vs. npz | Tries {TRIES}")

  plt.xscale("log")
  # plt.xticks(_x_items_, _x_items_, rotation = 45)

  plt.show()
