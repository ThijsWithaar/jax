import numpy as np
from jaxlib import xla_client
from . import _ndimage

for _name, _value in _ndimage.registrations().items():
  xla_client.register_custom_call_target(_name, _value, platform="cpu")

def affine_transform(ctx, input, htransform):
  dtype = ctx.get_shape(input).element_type()
  h_dtype = np.dtype('float32')

  i_shape = ctx.get_shape(input).dimensions()
  h_shape = ctx.get_shape(htransform).dimensions()
  assert(ctx.get_shape(htransform).element_type() == h_dtype)
  assert(h_shape == (3, 3)) # Require a homogeneous transform

  arr_shape_i = xla_client.Shape.array_shape(np.dtype(dtype), i_shape, tuple(range(len(i_shape) - 1, -1, -1)))
  arr_shape_h = xla_client.Shape.array_shape(h_dtype, h_shape, tuple(range(len(h_shape) - 1, -1, -1)))
  arr_shape_o = arr_shape_i

  descriptor_bytes = _ndimage.build_ndimage_descriptor(i_shape[1], i_shape[0])

  op_name = {
    np.dtype('float32')   : b"affine_transform_f32",
    np.dtype('float64')   : b"affine_transform_f64",
    np.dtype('complex64') : b"affine_transform_c64",
    np.dtype('complex128'): b"affine_transform_c128",
  }[dtype]

  return xla_client.ops.CustomCallWithLayout(
      ctx,
      op_name,
      operands=(
        xla_client.ops.Constant(ctx, np.frombuffer(descriptor_bytes, dtype=np.uint8)),
        input,
        htransform
      ),
      # Input shapes
      operand_shapes_with_layout=(
          xla_client.Shape.array_shape(np.dtype(np.uint8), (len(descriptor_bytes),), (0,)),
          arr_shape_i,
          arr_shape_h,
      ),
      # Output shapes
      shape_with_layout=arr_shape_o,
  )
