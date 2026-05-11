// psvr — Portable C++ core: kernel constructors (RBF, linear, polynomial).
//
// These functions are direct ports of the F6 Rcpp kernels in
// src/kernel_*.cpp. The pure-C++ inner loop (already long-double-
// accumulated in the F6 implementation) moves here; the Rcpp wrapping
// layer lives in src/binding_kernel.cpp.
//
// Memory: all output buffers are caller-allocated (column-major
// N1 × N2). For self-kernel calls (X1 == X2), pass the same pointer
// twice; the implementation computes the full matrix (no upper-
// triangle shortcut, matching the R baseline).
//
// Bit-identicality discipline (per F6 / paper TODO #8):
//   - RBF/linear/poly accumulate the inner reduction in `long double`
//     (matches R's LDOUBLE in `sum()`). A naive `double` accumulator
//     drifts at ~5e-17 for RBF, ~2e-15 for linear.
//   - polynomial degree >= 3 uses std::pow(long double, long double)
//     (matches R's powl via R_POW under HAVE_LONG_DOUBLE). degree == 2
//     uses the special-case base * base. Plain std::pow(double, double)
//     drifts at ~9e-13.

#ifndef PSVR_CORE_KERNEL_H
#define PSVR_CORE_KERNEL_H

#include "core_smo_types.h"

namespace psvr {

// RBF kernel: K(x, x') = exp(-||x - x'||^2 / (2 * sigma^2))
// X1 is N1 × p (row-major), X2 is N2 × p (row-major).
// Output `out` is N1 × N2 column-major (matches R's matrix layout).
void kernel_rbf(const double* X1, const double* X2,
                Index N1, Index N2, Index p, double sigma,
                double* out);

// Linear kernel: K(x, x') = x . x'
void kernel_linear(const double* X1, const double* X2,
                   Index N1, Index N2, Index p,
                   double* out);

// Polynomial kernel: K(x, x') = (x . x' + coef0)^degree
void kernel_poly(const double* X1, const double* X2,
                 Index N1, Index N2, Index p,
                 double coef0, int degree,
                 double* out);

}  // namespace psvr

#endif  // PSVR_CORE_KERNEL_H
