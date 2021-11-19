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
  I, H = primals
  dI, dH = tangents

  O = affine_transform(I, H)
  dIdy, dIdx = jnp.gradient(I)

  # With the chain-rule: dO/dH = dO/dc * dc/dH
  Y, X = jnp.mgrid[:I.shape[0], :I.shape[1]]
  one = jnp.ones_like(I)
  C = jnp.stack((Y, X, one))
  CH = np.tensordot(H, C, axes=1)

  # Derivative over the homogeneous projection, with the quotient-rule
  dc_dH = jnp.tensordot(dH, CH, axes=1)
  dxy_dH = dc_dH[:2,:,:] / jnp.tile(CH[2,:,:], (2,1,1))
  dxy_dH-= CH[:2,:,:] * jnp.tile(dc_dH[2,:,:], (2,1,1)) / jnp.tile(CH[2,:,:], (2,1,1))**2

  # The chain rule for H->(x,y)->O
  dO_dxy = jnp.stack((dIdy, dIdx))
  dOdH = (dO_dxy * dxy_dH).sum(axis=0)

  dOdI = affine_transform(dI, H)
  dO = dOdI + dOdH

  return O, dO

affine_transform_p = Primitive('affine_transform')
affine_transform_p.def_impl(affine_transform_impl)
affine_transform_p.def_abstract_eval(affine_transform_abstract_eval)
xla.register_translation(affine_transform_p, affine_transform_translation_rule_cpu, platform='cpu')
ad.primitive_jvps[affine_transform_p] = affine_transform_jvp_rule
