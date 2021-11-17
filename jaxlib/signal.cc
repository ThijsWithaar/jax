#include "jaxlib/kernel_pybind11_helpers.h"
#include "include/pybind11/pybind11.h"

namespace jax {


struct lfilter_descriptor
{
	size_t nB, nA, nX;
};

py::bytes BuildLfilterDescriptor(size_t nB, size_t nA, size_t nX)
{
	return PackDescriptor<lfilter_descriptor>({nB, nA, nX});
}

template<typename T>
void lfilter(void* out, void** in)
{
	const lfilter_descriptor s = UnpackDescriptor<lfilter_descriptor>(in[0]);
	const T* pb = in[1];
	const T* pa = in[2];
	const T* x = in[3];
	T* y = out[0];

	const nAB = std::max(s.nA, s.nB);

	// Zero-pad a,b. normalize by a[0]
	std::vector<T> a(nAB, T(0));
	T a0_inv = s.nA > 1 ? T(1)/pa[0] : T(1);	// This allows an empty a as input argument
	a[0] = T(1);
	for(size_t n=1; n < s.nA; n++)
		a[n] = pa[n] * a0_inv;

	std::vector<T> b(nAB, T(0));
	for(size_t n=0; n < s.nA; n++)
		b[n] = pb[n] * a0_inv;

	// State vector, extended to make the for-loop more regular
	std::vector<T> d(nAB + 1, T(0));

	// Direct mode II, like scipy.signal.lfilter
	// See https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.lfilter.html
	for(size_t n=0; n < s.nX; n++)
	{
		y[n] = b[0] * x[n] + d[0];
		for(size_t j=0; j < nAB-1; j++)
			d[j] = b[j+1]*x[n] - a[j+1]*y[n] + d[j+1];
	}
}


namespace {

pybind11::dict Registrations() {
  pybind11::dict dict;
  dict["lfilter"] = EncapsulateFunction(lfilter);
  return dict;
}

PYBIND11_MODULE(_signal, m)
{
	m.def("registrations", &Registrations);

	m.def("build_lfilter_descriptor", &BuildLfilterDescriptor);
}

}  // namespace
}  // namespace jax
