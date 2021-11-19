#include <algorithm>
#include <cstdint>
#include <complex>

#include "jaxlib/kernel_pybind11_helpers.h"
#include "include/pybind11/pybind11.h"

namespace py = pybind11;

namespace jax {
namespace {

template<typename T>
T clamp(T v, T lo, T hi)
{
	std::min(std::max(v, lo), hi);
}

template<typename T>
struct remove_complex
{
	using type = T;
};

template<typename T>
struct remove_complex<std::complex<T>>
{
	using type = T;
};

template<typename T>
struct coordinate
{
	T x,y;
};

struct ndimage_descriptor
{
	ptrdiff_t w, h;
};

py::bytes BuildNdImageDescriptor(ptrdiff_t w, ptrdiff_t h)
{
	return PackDescriptor<ndimage_descriptor>({w, h});
}

// Homogeneous coordinate projection
coordinate<double> project(const float* pH, coordinate<ptrdiff_t> ci)
{
	std::array<double, 3> o{0,0,0};
	for(size_t n=0; n<3; n++)
	{
		o[n] = ci.y*pH[0] + ci.x*pH[1] + pH[2];
		pH += 3;
	}
	return {o[1]/o[2], o[0]/o[2]};
}

// Bi-linear pixel interpolation
template<typename T>
T interp_linear(const T* pI, const ndimage_descriptor& s, coordinate<double> c)
{
	using T_real = typename remove_complex<T>::type;

	auto fetch = [&](ptrdiff_t x, ptrdiff_t y)
	{
		return clamp<ptrdiff_t>(x, 0, s.w-1) + s.w*clamp<ptrdiff_t>(y, 0, s.h-1);
	};

	T_real yi, xi;
	const T_real xf = std::modf(c.x, &xi);
	const T_real yf = std::modf(c.y, &yi);
	
	return fetch(xi, yi);	// Debug: Return coordinates

	T vu = fetch(xi, yi  )*(1-xf) + fetch(xi+1, yi  )*xf;
	T vd = fetch(xi, yi+1)*(1-xf) + fetch(xi+1, yi+1)*xf;
	return vu*(1-yf) + vd*yf;
}

template<typename T>
void affine_transform(void* out, void** in)
{
	const ndimage_descriptor s = **UnpackDescriptor<ndimage_descriptor>(reinterpret_cast<const char*>(in[0]), sizeof(ndimage_descriptor));
	const T* pI = reinterpret_cast<const T*>(in[1]);
	const float* pH = reinterpret_cast<const float*>(in[2]);
	T* pO = reinterpret_cast<T*>(out);
	
	std::cout << "s = " << s.w << ", " << s.h << std::endl;
	std::cout << "H = " << std::endl;
	for(size_t m=0; m<3; m++)
	{
		for(size_t n=0; n<3; n++)
			std::cout << pH[n+3*m] << " ";
		std::cout << std::endl;
	}

	coordinate<ptrdiff_t> co;
	for(co.y=0; co.y < s.h; co.y++)
	{
		for(co.x=0; co.x < s.w; co.x++)
		{
			coordinate<double> ci = project(pH, co);
			pO[co.y*s.w + co.x] = interp_linear(pI, s, {co.x+1, co.y});
			//pO[co.y*s.w + co.x] = pI[co.y*s.w + co.x];
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
