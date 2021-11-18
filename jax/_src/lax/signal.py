from jax.core import Primitive
from jax.interpreters import ad
from jax.interpreters import xla
from jax._src.lib import signal

__all__ = [
  "lfilter",
  "lfilter_p",
]

def lfilter(b, a, x):
  return lfilter_p.bind(b, a, x)

def lfilter_impl(b, a, x, axis=-1, zi=None):
  return xla.apply_primitive(lfilter_p, b, a, x)

def lfilter_abstract_eval(b, a, x):
  return x.update(shape=x.shape, dtype=x.dtype)

def _lfilter_translation_rule_cpu(ctx, avals_in, avals_out, b, a, x):
  return [signal.lfilter(ctx.builder, b, a, x)]

def _lfilter_jvp_rule(primals, tangents):
  b, a, x = primals
  db, da, dx = tangents

  y = lfilter(b, a, x)

  # The derivatives are analogues to their Fourier-domain counterparts,
  # as described on https://dsp.stackexchange.com/a/59718:
  # H = B / A
  # dH/dB =  1 / A
  # dH/dA = -H / A
  dy_db = lfilter(db, a, x)

  x_da  = lfilter(da, a, x)
  dy_da = lfilter(b , a, -x_da)

  dy_dx = lfilter(b , a, dx)
  dy = dy_db + dy_da + dy_dx

  return (y,), (dy,)

def _lfilter_batching_rule(vector_arg_values, batch_axes):
  res = lfilter(*vector_arg_values)
  return res, batch_axes[0]

lfilter_p = Primitive('lfilter')
lfilter_p.def_impl(lfilter_impl)
lfilter_p.def_abstract_eval(lfilter_abstract_eval)
if signal:
  xla.register_translation(lfilter_p, _lfilter_translation_rule_cpu, platform='cpu')
ad.primitive_jvps[lfilter_p] = _lfilter_jvp_rule
batching.primitive_batchers[lfilter_p] = _lfilter_batching_rule
