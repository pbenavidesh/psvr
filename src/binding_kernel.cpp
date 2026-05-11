// psvr — Rcpp binding layer for the core kernel functions.
//
// Thin wrappers that:
//   1. Accept Rcpp::NumericMatrix from R.
//   2. Validate the inputs (matching the pre-F7-C-full error messages
//      so behaviour is unchanged for callers of kernel_*_cpp).
//   3. Allocate the output NumericMatrix (which lives on R's side of
//      the heap and gets returned without a copy).
//   4. Call into psvr::kernel_* with raw double* pointers.
//
// Replaces src/kernel_rbf.cpp, src/kernel_linear.cpp, src/kernel_poly.cpp.
// The pure-C++ inner loops now live in src/core_kernel_*.cpp; the
// long-double accumulator and powl discipline are preserved there.

#include <Rcpp.h>
#include "core_kernel.h"

using Rcpp::NumericMatrix;
using Rcpp::stop;

// [[Rcpp::export]]
NumericMatrix kernel_rbf_cpp(const NumericMatrix& X1,
                              const NumericMatrix& X2,
                              double sigma) {
  const int N1 = X1.nrow();
  const int N2 = X2.nrow();
  const int p  = X1.ncol();
  if (X2.ncol() != p) stop("X1 and X2 must have the same number of columns");
  if (sigma <= 0.0)    stop("`sigma` must be positive");

  NumericMatrix K(N1, N2);
  psvr::kernel_rbf(REAL(X1), REAL(X2),
                   static_cast<psvr::Index>(N1),
                   static_cast<psvr::Index>(N2),
                   static_cast<psvr::Index>(p),
                   sigma, REAL(K));
  return K;
}

// [[Rcpp::export]]
NumericMatrix kernel_linear_cpp(const NumericMatrix& X1,
                                 const NumericMatrix& X2) {
  const int N1 = X1.nrow();
  const int N2 = X2.nrow();
  const int p  = X1.ncol();
  if (X2.ncol() != p) stop("X1 and X2 must have the same number of columns");

  NumericMatrix K(N1, N2);
  psvr::kernel_linear(REAL(X1), REAL(X2),
                      static_cast<psvr::Index>(N1),
                      static_cast<psvr::Index>(N2),
                      static_cast<psvr::Index>(p),
                      REAL(K));
  return K;
}

// [[Rcpp::export]]
NumericMatrix kernel_poly_cpp(const NumericMatrix& X1,
                               const NumericMatrix& X2,
                               double coef0,
                               int degree) {
  const int N1 = X1.nrow();
  const int N2 = X2.nrow();
  const int p  = X1.ncol();
  if (X2.ncol() != p) stop("X1 and X2 must have the same number of columns");
  if (degree < 1)      stop("`degree` must be >= 1");

  NumericMatrix K(N1, N2);
  psvr::kernel_poly(REAL(X1), REAL(X2),
                    static_cast<psvr::Index>(N1),
                    static_cast<psvr::Index>(N2),
                    static_cast<psvr::Index>(p),
                    coef0, degree, REAL(K));
  return K;
}
