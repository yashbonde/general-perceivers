import numpy as np
from data_pb2 import NDArray
from common import array_to_buffer, buffer_to_array

x = np.random.randn(1, 2, 3)
print(x)


np_proto = array_to_buffer(x)

# # serialisation
# _np_proto = NDArray()
# _np_proto.ParseFromString(np_proto.SerializeToString())
# np_proto = _np_proto
# # deserialisation

y = buffer_to_array(np_proto)

print(x == y)
