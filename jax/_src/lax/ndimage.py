from jax.core import Primitive
from jax.interpreters import ad
from jax.interpreters import xla
from jax._src.lib import ndimage


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

affine_transform_p = Primitive('affine_transform')
affine_transform_p.def_impl(affine_transform_impl)
affine_transform_p.def_abstract_eval(affine_transform_abstract_eval)
xla.register_translation(affine_transform_p, affine_transform_translation_rule_cpu, platform='cpu')
