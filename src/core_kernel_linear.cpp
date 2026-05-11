// psvr — Portable C++ core: linear kernel.
//
// Pure-C++ port of src/kernel_linear.cpp (F6). See core_kernel_rbf.cpp
// for the long-double accumulator rationale.

#include "core_kernel.h"

namespace psvr {

void kernel_linear(const double* X1, const double* X2,
                   Index N1, Index N2, Index p,
                   double* out) {
  // Match F6 traversal: outer i, inner j, innermost k. Column-major
  // input (R's default), column-major output.
  for (Index i = 0; i < N1; ++i) {
    for (Index j = 0; j < N2; ++j) {
      long double dot_ld = 0.0L;
      for (Index k = 0; k < p; ++k) {
        const double pp = X1[k * N1 + i] * X2[k * N2 + j];
        dot_ld += static_cast<long double>(pp);
      }
      out[j * N1 + i] = static_cast<double>(dot_ld);
    }
  }
}

}  // namespace psvr
