"""
See also:
https://github.com/danieljtait/jax_xla_adventures/blob/master/pybind11_register_custom_call/test.py
"""

import numpy as np
from jaxlib import xla_client
from . import _signal

for _name, _value in _signal.registrations().items():
  xla_client.register_custom_call_target(_name, _value, platform="cpu")

def lfilter(ctx, b, a, x):
  dtype = ctx.get_shape(x).element_type()

  b_shape = ctx.get_shape(b).dimensions()
  a_shape = ctx.get_shape(a).dimensions()
  x_shape = ctx.get_shape(x).dimensions()

  n = len(x_shape)
  arr_shape_b = xla_client.Shape.array_shape(np.dtype(dtype), b_shape, tuple(range(len(b_shape) - 1, -1, -1)))
  arr_shape_a = xla_client.Shape.array_shape(np.dtype(dtype), a_shape, tuple(range(len(a_shape) - 1, -1, -1)))
  arr_shape_x = xla_client.Shape.array_shape(np.dtype(dtype), x_shape, tuple(range(len(x_shape) - 1, -1, -1)))

  descriptor_bytes = _signal.build_lfilter_descriptor(b_shape[0], a_shape[0], x_shape[0])

  op_name = {
    np.dtype('float32')   : b"lfilter_f32",
    np.dtype('float64')   : b"lfilter_f64",
    np.dtype('complex64') : b"lfilter_c64",
    np.dtype('complex128'): b"lfilter_c128",
  }[dtype]

  return xla_client.ops.CustomCallWithLayout(
      ctx,
      op_name,
      operands=(
        xla_client.ops.Constant(ctx, np.frombuffer(descriptor_bytes, dtype=np.uint8)),
        b, a, x
      ),
      # Input shapes
      operand_shapes_with_layout=(
          xla_client.Shape.array_shape(np.dtype(np.uint8), (len(descriptor_bytes),), (0,)),
          arr_shape_b,
          arr_shape_a,
          arr_shape_x,
      ),
      # Output shapes
      shape_with_layout=arr_shape_x,
  )
