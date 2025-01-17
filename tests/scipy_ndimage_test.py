# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from functools import partial

import numpy as np

from absl.testing import absltest
from absl.testing import parameterized
import scipy.ndimage as osp_ndimage

from jax import grad, jacfwd, jvp
from jax._src import test_util as jtu
from jax import dtypes
from jax.scipy import ndimage as lsp_ndimage
from jax._src.util import prod

from jax.config import config
config.parse_flags_with_absl()


float_dtypes = jtu.dtypes.floating
int_dtypes = jtu.dtypes.integer


def _fixed_ref_map_coordinates(input, coordinates, order, mode, cval=0.0):
  # SciPy's implementation of map_coordinates handles boundaries incorrectly,
  # unless mode='reflect'. For order=1, this only affects interpolation outside
  # the bounds of the original array.
  # https://github.com/scipy/scipy/issues/2640
  assert order <= 1
  padding = [(max(-np.floor(c.min()).astype(int) + 1, 0),
              max(np.ceil(c.max()).astype(int) + 1 - size, 0))
             for c, size in zip(coordinates, input.shape)]
  shifted_coords = [c + p[0] for p, c in zip(padding, coordinates)]
  pad_mode = {
      'nearest': 'edge', 'mirror': 'reflect', 'reflect': 'symmetric'
  }.get(mode, mode)
  if mode == 'constant':
    padded = np.pad(input, padding, mode=pad_mode, constant_values=cval)
  else:
    padded = np.pad(input, padding, mode=pad_mode)
  result = osp_ndimage.map_coordinates(
      padded, shifted_coords, order=order, mode=mode, cval=cval)
  return result


@jtu.with_config(jax_numpy_rank_promotion="raise")
class NdimageTest(jtu.JaxTestCase):
  def testAffineTransform(self):
    osp_op = partial(osp_ndimage.affine_transform, order=1, mode='nearest', cval=0)
    lsp_op = partial(lsp_ndimage.affine_transform, order=1, mode='nearest', cval=0)

    I = np.zeros((5,7))
    I[2:4, 3] = 1.
    I[2, 4] = 2.

    # Translation
    M = np.array([[1, 0], [0, 1]], dtype=np.float32)
    O = np.array([1.1, 2.8], dtype=np.float32)
    args_maker = lambda: [I, M, O]
    self._CheckAgainstNumpy(osp_op, lsp_op, args_maker, tol=1e-6)

    # Rotation
    M = np.array([[.9, .1], [-.1, 9]], dtype=np.float32)
    O = np.array([-.1, +.2], dtype=np.float32)
    args_maker = lambda: [I, M, O]
    self._CheckAgainstNumpy(osp_op, lsp_op, args_maker, tol=1e-6)

  def testAffineTransform_JVP(self):
    Y, X = np.mgrid[:5, :6]
    alpha = .02
    x0 = 3.; y0 = 2.
    #G = 9 * np.exp(-alpha * ((X-x0)**2 + (Y-y0)**2)).astype(np.float32)
    #import pdb; pdb.set_trace()
    G = (1.3*np.power(np.abs(X - x0), 1.2) + 0.0*(Y-y0)).astype(np.float32)
    print("G = ", G.round(2))

    H = np.array([
        [1.2, 0  , 0],
        [0  , .75, 0],
        [0  , 0  , 1.5]
    ], dtype=np.float32)

    def jvp_num(dH):
        eps = 1e-2
        dy = lsp_ndimage.affine_transform(G, H + eps*dH) - lsp_ndimage.affine_transform(G, H - eps*dH)
        return dy / (2*eps)

    for m in range(2,3):
        for n in range(3):
            dH = np.zeros_like(H)
            dH[m,n] = 1
            _, df_dH = jvp(lsp_ndimage.affine_transform, (G, H), (0*G, dH))
            df_dH_num = jvp_num(dH)

            self.assertAllClose(df_dH, df_dH_num, atol=2)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}_coordinates={}_order={}_mode={}_cval={}_impl={}_round={}".format(
          jtu.format_shape_dtype_string(shape, dtype),
          jtu.format_shape_dtype_string(coords_shape, coords_dtype),
          order, mode, cval, impl, round_),
       "rng_factory": rng_factory, "shape": shape,
       "coords_shape": coords_shape, "dtype": dtype,
       "coords_dtype": coords_dtype, "order": order, "mode": mode,
       "cval": cval, "impl": impl, "round_": round_}
      for shape in [(5,), (3, 4), (3, 4, 5)]
      for coords_shape in [(7,), (2, 3, 4)]
      for dtype in float_dtypes + int_dtypes
      for coords_dtype in float_dtypes
      for order in [0, 1]
      for mode in ['wrap', 'constant', 'nearest', 'mirror', 'reflect']
      for cval in ([0, -1] if mode == 'constant' else [0])
      for impl, rng_factory in [
          ("original", partial(jtu.rand_uniform, low=0, high=1)),
          ("fixed", partial(jtu.rand_uniform, low=-0.75, high=1.75)),
      ]
      for round_ in [True, False]))
  def testMapCoordinates(self, shape, dtype, coords_shape, coords_dtype, order,
                         mode, cval, impl, round_, rng_factory):

    def args_maker():
      x = np.arange(prod(shape), dtype=dtype).reshape(shape)
      coords = [(size - 1) * rng(coords_shape, coords_dtype) for size in shape]
      if round_:
        coords = [c.round().astype(int) for c in coords]
      return x, coords

    rng = rng_factory(self.rng())
    lsp_op = lambda x, c: lsp_ndimage.map_coordinates(
        x, c, order=order, mode=mode, cval=cval)
    impl_fun = (osp_ndimage.map_coordinates if impl == "original"
                else _fixed_ref_map_coordinates)
    osp_op = lambda x, c: impl_fun(x, c, order=order, mode=mode, cval=cval)
    if dtype in float_dtypes:
      epsilon = max([dtypes.finfo(dtypes.canonicalize_dtype(d)).eps
                     for d in [dtype, coords_dtype]])
      self._CheckAgainstNumpy(osp_op, lsp_op, args_maker, tol=100*epsilon)
    else:
      self._CheckAgainstNumpy(osp_op, lsp_op, args_maker, tol=0)

  def testMapCoordinatesErrors(self):
    x = np.arange(5.0)
    c = [np.linspace(0, 5, num=3)]
    with self.assertRaisesRegex(NotImplementedError, 'requires order<=1'):
      lsp_ndimage.map_coordinates(x, c, order=2)
    with self.assertRaisesRegex(
        NotImplementedError, 'does not yet support mode'):
      lsp_ndimage.map_coordinates(x, c, order=1, mode='grid-wrap')
    with self.assertRaisesRegex(ValueError, 'sequence of length'):
      lsp_ndimage.map_coordinates(x, [c, c], order=1)

  def testMapCoordinateDocstring(self):
    self.assertIn("Only nearest neighbor",
                  lsp_ndimage.map_coordinates.__doc__)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}_order={}".format(np.dtype(dtype), order),
       "dtype": dtype, "order": order}
      for dtype in float_dtypes + int_dtypes
      for order in [0, 1]))
  def testMapCoordinatesRoundHalf(self, dtype, order):
    x = np.arange(-3, 3, dtype=dtype)
    c = np.array([[.5, 1.5, 2.5, 3.5]])
    def args_maker():
      return x, c

    lsp_op = lambda x, c: lsp_ndimage.map_coordinates(x, c, order=order)
    osp_op = lambda x, c: osp_ndimage.map_coordinates(x, c, order=order)
    self._CheckAgainstNumpy(osp_op, lsp_op, args_maker)

  def testContinuousGradients(self):
    # regression test for https://github.com/google/jax/issues/3024

    def loss(delta):
      x = np.arange(100.0)
      border = 10
      indices = np.arange(x.size) + delta
      # linear interpolation of the linear function y=x should be exact
      shifted = lsp_ndimage.map_coordinates(x, [indices], order=1)
      return ((x - shifted) ** 2)[border:-border].mean()

    # analytical gradient of (x - (x - delta)) ** 2 is 2 * delta
    self.assertAllClose(grad(loss)(0.5), 1.0, check_dtypes=False)
    self.assertAllClose(grad(loss)(1.0), 2.0, check_dtypes=False)


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
