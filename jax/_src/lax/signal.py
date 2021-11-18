import numpy as np

from jax.core import Primitive
from jax.interpreters import ad
from jax.interpreters import xla
from jax._src.numpy import lax_numpy as jnp
from jax._src.numpy import fft
from jax._src.lib import xla_client
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

  # Calculate derivatives in the Fourier domain, then do ifft
  # https://dsp.stackexchange.com/a/59718/627
  polyval = np.polynomial.polynomial.polyval
  w = jnp.linspace(0, np.pi, x.size, endpoint=False)
  z = jnp.exp(-1j * w)
  B = polyval(z, b, tensor=False)
  A = polyval(z, a, tensor=False)
  H = B/A

  za = z.reshape(-1,1) ** jnp.r_[0:a.size].reshape(1,-1)
  Ja = (-H/A).reshape(-1,1)* za

  zb = z.reshape(-1,1) ** jnp.r_[0:b.size].reshape(1,-1)
  Jb = (1/A).reshape(-1,1) * zb

  dy_dx = lfilter(b, a, dx)
  dy = dy_dx + fft.ifft(Ja @ da + Jb @ db, axis=0).real

  import pdb; pdb.set_trace()
  return (y,), (dy,)

lfilter_p = Primitive('lfilter')
lfilter_p.def_impl(lfilter_impl)
lfilter_p.def_abstract_eval(lfilter_abstract_eval)
if signal:
  xla.register_translation(lfilter_p, _lfilter_translation_rule_cpu, platform='cpu')
ad.primitive_jvps[lfilter_p] = _lfilter_jvp_rule

#xla.register_translation(lfilter_p, _lfilter_translation_rule)
#ad.deflinear2(fft_p, fft_transpose_rule)
#batching.primitive_batchers[lfilter_p] = lfilter_batching_rule
