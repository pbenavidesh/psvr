#include <Rcpp.h>
using namespace Rcpp;

// Matches R's `sum(xi * xj)` exactly:
//   tmp    <- xi * xj                (vectorized double arithmetic)
//   result <- sum(tmp)               (R's sum() uses long-double accumulator)
//
// We accumulate in `long double` to mirror R's summary.c LDOUBLE
// accumulator, matching the bit pattern of R's nested-loop reference.
//
// [[Rcpp::export]]
NumericMatrix kernel_linear_cpp(const NumericMatrix& X1,
                                 const NumericMatrix& X2) {
  const int N1 = X1.nrow();
  const int N2 = X2.nrow();
  const int p  = X1.ncol();
  if (X2.ncol() != p) {
    stop("X1 and X2 must have the same number of columns");
  }

  NumericMatrix K(N1, N2);

  for (int i = 0; i < N1; i++) {
    for (int j = 0; j < N2; j++) {
      long double dot_ld = 0.0L;
      for (int k = 0; k < p; k++) {
        const double pp = X1(i, k) * X2(j, k);
        dot_ld += static_cast<long double>(pp);
      }
      K(i, j) = static_cast<double>(dot_ld);
    }
  }
  return K;
}
