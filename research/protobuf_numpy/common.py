import numpy as np
from data_pb2 import NDArray

def array_to_buffer(x: np.ndarray):
  # Getting array from buffer is so simple, should I just remove
  # the complexity from the proto? But it's not that much so let
  # it be
  dtype_byte_order = {
    "<": NDArray.DType.LITTLE_ENDIAN,
    ">": NDArray.DType.BIG_ENDIAN,
    "=": NDArray.DType.NATIVE,
    "|": NDArray.DType.NA,
  }

  data = NDArray(
    shape = list(x.shape),
    dtype = NDArray.DType(
      type = x.dtype.name,
      byte_order = dtype_byte_order[x.dtype.byteorder],
      names = x.dtype.names,
      fields = x.dtype.fields,
    ),
    data = x.tobytes(),
    strides=list(x.strides),
  )

  if x.dtype.subdtype:
    data.dtype.CopyFrom(
      NDArray.DType.SubDType(
        type = x.dtype.subdtype[0].name,
        names = x.dtype.subdtype[1],
      )
    )
  
  return data


def buffer_to_array(x: NDArray):
  # Though now irrelevant this comment is here for reading
  # https://github.com/numpy/numpy/issues/12661#issuecomment-451455761
  # `np.empty`` just means "create an array, I don't care what values it
  # contains because I'm going to replace all of them"

  out = np.frombuffer(x.data, dtype = np.dtype(NDArray.DType.Type.keys()[x.dtype.type])).reshape(x.shape)
  return out