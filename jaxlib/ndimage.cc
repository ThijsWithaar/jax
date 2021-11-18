#include <cstdint>
#include <complex>

#include "jaxlib/kernel_pybind11_helpers.h"
#include "include/pybind11/pybind11.h"

namespace py = pybind11;

namespace jax {
namespace {

template<typename T>
struct coordinate
{
	T x,y;
};

struct ndimage_descriptor
{
	ptrdiff_t h, w;
};

py::bytes BuildNdImageDescriptor(ptrdiff_t h, ptrdiff_t w)
{
	return PackDescriptor<ndimage_descriptor>({h, w});
}

// Homogeneous coordinate projection
coordinate<double> project(const double* pH, coordinate<ptrdiff_t> ci)
{
	std::array<double> o{0,0,0};
	for(size_t n=0; n<3; n++)
	{
		o[n] = ci.x*pH[0] + ci.x*pH[1] + pH[2];
		pH += 3;
	}
	return {o[0]/o[2], o[1]/o[2]};
}

// Bi-linear pixel interpolation
template<typename T>
T interp_linear(const T* pI, const ndimage_descriptor& s, coordinate<double> c)
{
	auto fetch = [](ptrdiff_t x, ptrdiff_t y)
	{
		return std::clamp(x, 0, s.w-1) + s.w*std::clamp(y, 0, s.h-1);
	};

	double yi, xi;
	const double xf = std::modf(c.x, &xi);
	const double yf = std::modf(c.y, &yi);

	T vu = fetch(xi, yi  )*(1-xf) + fetch(xi+1, yi  )*xf;
	T vd = fetch(xi, yi+1)*(1-xf) + fetch(xi+1, yi+1)*xf;
	return vu*(1-yf) + vd*yf;
}

template<typename T>
void affine_transform(void* out, void** in)
{
	const ndimage_descriptor s = **UnpackDescriptor<ndimage_descriptor>(reinterpret_cast<const char*>(in[0]), sizeof(ndimage_descriptor));
	const T* pI = reinterpret_cast<const T*>(in[1]);
	const double* pH = reinterpret_cast<const double*>(in[2]);
	T* pO = reinterpret_cast<T*>(out);

	coordinate<ptrdiff_t> co{0,0};
	for(co[0]=0; co[0] < h; co[0]++)
	{
		for(co[1]=0; co[1] < w; co[1]++)
		{
			coordinate<double> ci = project(pH, co);
			pO[y*w + x] = interp_linear(pI, ndimage_descriptor, ci);
		}
	}
}

pybind11::dict Registrations() {
  pybind11::dict dict;
  dict["affine_transform_f32" ] = EncapsulateFunction(affine_transform<float>);
  dict["affine_transform_f64" ] = EncapsulateFunction(affine_transform<double>);
  dict["affine_transform_c64" ] = EncapsulateFunction(affine_transform<std::complex<float>>);
  dict["affine_transform_c128"] = EncapsulateFunction(affine_transform<std::complex<double>>);
  return dict;
}

PYBIND11_MODULE(_ndimage, m)
{
	m.def("registrations", &Registrations);

	m.def("build_ndimage_descriptor", &BuildNdImageDescriptor);
}

}  // namespace
}  // namespace jax
