#include <Rcpp.h>
#include <cmath>
using namespace Rcpp;

// Bit-identicality with R's nested-loop kernel_matrix() in R/kernel.R
// requires matching R's operation pipeline exactly:
//   sigma2 <- 2 * sigma^2            (R's R_POW special case: y==2 -> x*x)
//   d      <- xi - xj                (vectorized double arithmetic)
//   tmp    <- d * d                  (vectorized double arithmetic)
//   s      <- sum(tmp)               (R's sum() uses long-double accumulator)
//   result <- exp(-s / sigma2)
//
// The critical subtlety: R's sum() in src/main/summary.c uses
//   `LDOUBLE s = 0.0;  for (i ...) s += x[i];`
// (long double on Windows/mingw-w64 is 80-bit extended precision). A plain
// `double` accumulator drifts at machine epsilon scale, breaking snapshots.
// We therefore accumulate the sum-of-squared-differences in `long double`
// and cast back to `double` only at the end, mirroring R's behaviour.
//
// [[Rcpp::export]]
NumericMatrix kernel_rbf_cpp(const NumericMatrix& X1,
                              const NumericMatrix& X2,
                              double sigma) {
  const int N1 = X1.nrow();
  const int N2 = X2.nrow();
  const int p  = X1.ncol();
  if (X2.ncol() != p) {
    stop("X1 and X2 must have the same number of columns");
  }
  if (sigma <= 0.0) {
    stop("`sigma` must be positive");
  }

  const double sigma_sq = sigma * sigma;
  const double sigma2   = 2.0 * sigma_sq;

  NumericMatrix K(N1, N2);

  for (int i = 0; i < N1; i++) {
    for (int j = 0; j < N2; j++) {
      long double sq_dist_ld = 0.0L;
      for (int k = 0; k < p; k++) {
        const double d  = X1(i, k) - X2(j, k);
        const double dd = d * d;
        sq_dist_ld += static_cast<long double>(dd);
      }
      const double sq_dist = static_cast<double>(sq_dist_ld);
      K(i, j) = std::exp(-sq_dist / sigma2);
    }
  }
  return K;
}
