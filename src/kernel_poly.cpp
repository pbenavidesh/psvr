#include <Rcpp.h>
#include <cmath>
using namespace Rcpp;

// Matches R's `(sum(xi * xj) + coef0)^degree` exactly:
//   tmp    <- xi * xj                (vectorized double)
//   dot    <- sum(tmp)               (R's sum() uses long-double accumulator)
//   base   <- dot + coef0            (double)
//   result <- base ^ degree          (R_POW: special case y==2 -> x*x; else pow)
//
// Long-double accumulator mirrors R's summary.c (see kernel_rbf.cpp for
// the full rationale). For the power step we mirror R_POW exactly:
//   degree == 1: base
//   degree == 2: base * base                                        (R_POW special case)
//   degree >= 3: (double) powl((long double) base, (long double) degree)
// The last form matches R's `(double) powl((LDOUBLE) x, (LDOUBLE) y)`
// in arithmetic.c under HAVE_LONG_DOUBLE. A plain std::pow(double, double)
// drifts at ~1e-13 relative to R for cubic exponents — enough to perturb
// SMO trajectories over many iterations and break the polynomial snapshots.
//
// [[Rcpp::export]]
NumericMatrix kernel_poly_cpp(const NumericMatrix& X1,
                               const NumericMatrix& X2,
                               double coef0,
                               int degree) {
  const int N1 = X1.nrow();
  const int N2 = X2.nrow();
  const int p  = X1.ncol();
  if (X2.ncol() != p) {
    stop("X1 and X2 must have the same number of columns");
  }
  if (degree < 1) {
    stop("`degree` must be >= 1");
  }

  NumericMatrix K(N1, N2);
  const long double deg_ld = static_cast<long double>(degree);

  for (int i = 0; i < N1; i++) {
    for (int j = 0; j < N2; j++) {
      long double dot_ld = 0.0L;
      for (int k = 0; k < p; k++) {
        const double pp = X1(i, k) * X2(j, k);
        dot_ld += static_cast<long double>(pp);
      }
      const double dot  = static_cast<double>(dot_ld);
      const double base = dot + coef0;
      if (degree == 1) {
        K(i, j) = base;
      } else if (degree == 2) {
        K(i, j) = base * base;
      } else {
        K(i, j) = static_cast<double>(
                    std::pow(static_cast<long double>(base), deg_ld));
      }
    }
  }
  return K;
}
