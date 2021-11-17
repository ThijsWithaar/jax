"""
See also:
https://github.com/danieljtait/jax_xla_adventures/blob/master/pybind11_register_custom_call/test.py
"""

from . import _signal

for _name, _value in _signal.registrations().items():
  xla_client.register_custom_call_target(_name, _value, platform="cpu")

def lfilter(ctx, b, a, x):
  descriptor_bytes = _signal.build_lfilter_descriptor(b.size, a.size, x.size)

  return xla_client.ops.CustomCallWithLayout(
      ctx,
      b"lfilter",
      operands=(b, a, x),
      shape_with_layout=xla_client.Shape.array_shape(jnp.dtype(jnp.complex64), (), ()),
      operand_shapes_with_layout=(
          xla_client.ops.Constant(ctx, np.frombuffer(descriptor_bytes, dtype=np.uint8)),
          xla_client.Shape.array_shape(dtype, b_shape.dimensions(), layout),
          xla_client.Shape.array_shape(dtype, a_shape.dimensions(), layout),
          xla_client.Shape.array_shape(dtype, x_shape.dimensions(), layout),
      )
  )
