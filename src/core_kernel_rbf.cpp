// psvr — Portable C++ core: RBF kernel.
//
// Pure-C++ port of src/kernel_rbf.cpp's inner loop (F6). The Rcpp-
// facing wrapper lives in src/binding_kernel.cpp.
//
// Bit-identicality with R's nested-loop kernel_matrix() requires:
//   sigma2 = 2 * sigma^2          (R_POW special case: y==2 -> x*x)
//   d      = xi - xj              (double subtraction)
//   tmp    = d * d                (double multiplication)
//   s      = sum(tmp)             (long-double accumulator — see below)
//   out    = exp(-s / sigma2)
//
// Long-double accumulator: R's sum() in src/main/summary.c declares
// LDOUBLE accumulators which become 80-bit extended precision on x87
// (mingw-w64 / Rtools). A plain `double` accumulator drifts at ~5e-17
// per output entry, breaking snapshots.
//
// Memory layout: X1 (N1 × p) and X2 (N2 × p) are column-major (R's
// default). Output is also column-major N1 × N2.

#include <cmath>
#include "core_kernel.h"

namespace psvr {

void kernel_rbf(const double* X1, const double* X2,
                Index N1, Index N2, Index p, double sigma,
                double* out) {
  const double sigma2 = 2.0 * sigma * sigma;
  // Match F6 traversal order exactly: outer i, inner j, innermost k.
  // R's matrix indexing X1(i, k) is X1[k * N1 + i] in column-major.
  for (Index i = 0; i < N1; ++i) {
    for (Index j = 0; j < N2; ++j) {
      long double sq_dist_ld = 0.0L;
      for (Index k = 0; k < p; ++k) {
        const double d  = X1[k * N1 + i] - X2[k * N2 + j];
        const double dd = d * d;
        sq_dist_ld += static_cast<long double>(dd);
      }
      const double sq_dist = static_cast<double>(sq_dist_ld);
      out[j * N1 + i] = std::exp(-sq_dist / sigma2);
    }
  }
}

}  // namespace psvr
