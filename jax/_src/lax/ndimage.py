import numpy as np

from jax.core import Primitive
from jax.interpreters import ad
from jax.interpreters import xla
from jax._src.lib import ndimage
from jax._src.numpy import lax_numpy as jnp


__all__ = [
  "affine_transform",
  "affine_transform_p",
]

def affine_transform(I, H):
  return affine_transform_p.bind(I, H)

def affine_transform_impl(I, H):
  return xla.apply_primitive(affine_transform_p, I, H)

def affine_transform_abstract_eval(I, H):
  return I.update(shape=I.shape, dtype=I.dtype)

def affine_transform_translation_rule_cpu(ctx, avals_in, avals_out, I, H):
  return [ndimage.affine_transform(ctx.builder, I, H)]

def affine_transform_jvp_rule(primals, tangents):
    """
    import sympy
    sympy.init_printing(use_latex=False, use_unicode=False)

    x_i, y_i = sympy.symbols('x_i y_i')
    c_i = sympy.Matrix([y_i, x_i, 1])
    H = sympy.MatrixSymbol('H', 3,3)

    c_oh = H @ c_i
    c_o = [c_oh[0,0] / c_oh[2,0], c_oh[1,0] / c_oh[2,0]]

    J = []
    for m in range(3):
         for n in range(3):
             J.append([sympy.diff(c_oi, H[m,n]) for c_oi in c_o])
             sympy.pprint(J[-1])

    """
    I, H = primals
    dI, dH = tangents

    O = affine_transform(I, H)

    # Split into image gradients dI/dxy and transform gradients dxy/dH
    dIdy, dIdx = jnp.gradient(I)
    dI_dxy = jnp.stack((dIdy, dIdx))

    # Input coordinates
    Y, X = jnp.mgrid[:I.shape[0], :I.shape[1]]
    one = jnp.ones_like(I)
    yx1 = jnp.stack((Y, X, one))

    # Should be 3 x I.shape
    c_i = np.tensordot(H, yx1, axes=1)

    # For debug:
    J00x = Y / c_i[2,:,:]
    J01x = X / c_i[2,:,:]
    J02x = 1 / c_i[2,:,:]
    Z = jnp.zeros_like(J00x)
    # J1_y = J0_x
    den0 = c_i[0,:,:] / c_i[2,:,:]**2
    den1 = c_i[1,:,:] / c_i[2,:,:]**2
    J20x = -Y * den0
    J20y = -Y * den1
    J21x = -X * den0
    J21y = -X * den1
    J22x = -1 * den0
    J22y = -1 * den1

    def s2(x,y):
        return jnp.stack((x,y))

    def s3(x,y,z):
        return jnp.stack((x,y,z))

    dOdc = jnp.stack((
        s3(s2(J00x,    Z), s2(J01x,    Z), s2(J02x,    Z)),
        s3(s2(   Z, J00x), s2(   Z, J01x), s2(   Z, J02x)),
        s3(s2(J20x, J20y), s2(J21x, J21y), s2(J22x, J22y)),
    ))
    #dOdc = np.swapaxes(dOdc, 0, 1)
    dOdH = np.tensordot(dH, dOdc, axes=2)

    # Should be 2 x 3 x I.shape
    #c1_i = np.tile(c_i[2,:,:], (2,3,1,1))
    #cxy_i = np.swapaxes(np.tile(c_i[:2,:,:],(3,1,1,1)),0,1)

    # These should each be 2 x 3 x I.shape
    #z = jnp.zeros_like(yx1)
    #dH_dy = np.stack((yx1,z)) / c1_i
    #dH_dx = np.stack((z,yx1)) / c1_i
    #dH_d1 = -np.tile(yx1,(2,1,1,1)) * cxy_i / c1_i**2

    # Should be 3 x 2 x 3 x I.shape
    #dH_dc = jnp.stack((dH_dy, dH_dx, dH_d1))
    #dOdH  = jnp.tensordot(dH, dH_dc, axes=[(1,0),(2,0)])
    dI_dc = (dI_dxy * dOdH).sum(axis=0)

    # Combine derivatives of I and H
    dOdI = affine_transform(dI, H)
    dO = dOdI + dI_dc
    #import pdb; pdb.set_trace()

    return O, dO


affine_transform_p = Primitive('affine_transform')
affine_transform_p.def_impl(affine_transform_impl)
affine_transform_p.def_abstract_eval(affine_transform_abstract_eval)
xla.register_translation(affine_transform_p, affine_transform_translation_rule_cpu, platform='cpu')
ad.primitive_jvps[affine_transform_p] = affine_transform_jvp_rule
