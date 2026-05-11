// psvr — Portable C++ core: polynomial kernel.
//
// Pure-C++ port of src/kernel_poly.cpp (F6). See core_kernel_rbf.cpp
// for the long-double accumulator rationale. The degree >= 3 path
// uses std::pow(long double, long double) to match R's
// `(double) powl((LDOUBLE) x, (LDOUBLE) y)` (arithmetic.c under
// HAVE_LONG_DOUBLE). degree == 2 uses base * base (R_POW special
// case). Plain std::pow(double, double) drifts at ~1e-13.

#include <cmath>
#include "core_kernel.h"

namespace psvr {

void kernel_poly(const double* X1, const double* X2,
                 Index N1, Index N2, Index p,
                 double coef0, int degree,
                 double* out) {
  const long double deg_ld = static_cast<long double>(degree);

  for (Index i = 0; i < N1; ++i) {
    for (Index j = 0; j < N2; ++j) {
      long double dot_ld = 0.0L;
      for (Index k = 0; k < p; ++k) {
        const double pp = X1[k * N1 + i] * X2[k * N2 + j];
        dot_ld += static_cast<long double>(pp);
      }
      const double dot  = static_cast<double>(dot_ld);
      const double base = dot + coef0;
      double val;
      if (degree == 1) {
        val = base;
      } else if (degree == 2) {
        val = base * base;
      } else {
        val = static_cast<double>(
                std::pow(static_cast<long double>(base), deg_ld));
      }
      out[j * N1 + i] = val;
    }
  }
}

}  // namespace psvr
