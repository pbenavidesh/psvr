// psvr — Standalone smoke check for the portable C++ core.
//
// Compiles core_*.cpp WITHOUT R headers, verifying that the core
// itself has no R-isms. The dgemv_ symbol is stubbed below (naive
// implementation, NOT bit-identical to R's BLAS) so the standalone
// binary can link.
//
/*
 * Build (Windows / Rtools45 — PowerShell with PATH set to rtools45 bin):
 *   g++ -O2 -std=c++14 -DPSVR_STANDALONE_BUILD `
 *       -I src `
 *       dev/check_core.cpp `
 *       src/core_smo_solve.cpp src/core_block_k4.cpp `
 *       src/core_kernel_rbf.cpp src/core_kernel_linear.cpp `
 *       src/core_kernel_poly.cpp `
 *       -o dev/check_core.exe
 *
 * Build (Linux / macOS):
 *   g++ -O2 -std=c++14 -DPSVR_STANDALONE_BUILD -I src \
 *       dev/check_core.cpp src/core_*.cpp -o dev/check_core
 *
 * Expected output: a small fit summary; values WILL DIFFER from the
 * R build (BLAS implementation differs). Use this binary only as a
 * "does it compile and run" check, not for numerical validation.
 */

#include <cstdio>
#include <vector>
#include <cmath>

#include "core_smo_solve.h"
#include "core_smo_types.h"
#include "core_kernel.h"

// Standalone dgemv_ stub: naive matvec, prints a warning banner on
// first call. NOT BIT-IDENTICAL to R's BLAS — only here so the smoke
// binary links.
extern "C" {
void dgemv_(const char* trans, const int* m, const int* n,
            const double* alpha, const double* A, const int* lda,
            const double* x, const int* incx,
            const double* beta, double* y, const int* incy) {
  static bool warned = false;
  if (!warned) {
    std::fprintf(stderr,
      "========================================================================\n"
      "WARNING: standalone build uses ad-hoc dgemv. Numerical results may\n"
      "differ from R-build. Use R-build for bit-identicality verification.\n"
      "========================================================================\n");
    warned = true;
  }
  // Compute y := alpha * A * x + beta * y (trans = 'N', column-major).
  const int N = *m;
  const int Mc = *n;
  const double a = *alpha;
  const double b = *beta;
  const int incx_v = *incx;
  const int incy_v = *incy;
  (void)trans; (void)lda;
  for (int i = 0; i < N; ++i) {
    double acc = 0.0;
    for (int j = 0; j < Mc; ++j) {
      acc += A[j * N + i] * x[j * incx_v];
    }
    y[i * incy_v] = a * acc + b * y[i * incy_v];
  }
}
}

int main() {
  using namespace psvr;

  // Tiny fixture: N = 8 samples, p = 2 features, RBF sigma = 1.
  // y values strictly positive (MAPE assumption).
  const Index N = 8;
  const Index p = 2;
  const std::vector<double> X = {
    // column-major N×p: column 0 then column 1.
    -1.0, -0.5,  0.0,  0.5,  1.0, -0.7,  0.3,  0.9,    // feature 0
     0.5,  1.2, -0.3, -1.1,  0.7,  0.2,  0.9, -0.4     // feature 1
  };
  const std::vector<double> y = {
    1.2, 0.9, 1.0, 1.5, 0.8, 1.1, 1.3, 0.95
  };

  // Build Omega = RBF kernel matrix.
  std::vector<double> Omega(N * N);
  kernel_rbf(X.data(), X.data(), N, N, p, 1.0, Omega.data());
  // Jitter the diagonal (matches the R-side R/mape_svr.R)
  for (Index k = 0; k < N; ++k) Omega[k * N + k] += 1e-6;

  // Build options + fit.
  FitOptions opts;
  opts.C = 10.0;
  opts.eps = 5.0;
  opts.tol = 1e-5;
  opts.max_iter = 10000;
  opts.block_k4_enabled = true;

  FitResult res = smo_fit(Omega.data(), N, y.data(), opts);

  std::printf("\n=== standalone smoke (NOT bit-identical to R-build) ===\n");
  std::printf("converged           : %s\n", res.converged ? "TRUE" : "FALSE");
  std::printf("iterations          : %td\n", res.iterations);
  std::printf("b                   : %.10f\n", res.b);
  std::printf("joint_updates       : %td\n", res.joint_updates);
  std::printf("k2_fallbacks        : %td\n", res.k2_fallbacks);
  std::printf("decoupling_rate     : %.4f\n", res.decoupling_rate);
  std::printf("early_phase_decouple: %.4f\n", res.early_phase_decoupling_rate);
  std::printf("late_phase_decouple : %.4f\n", res.late_phase_decoupling_rate);
  std::printf("alpha[0..N-1]       :");
  for (Index k = 0; k < N; ++k) std::printf(" %+.4f", res.alpha[k]);
  std::printf("\n");
  std::printf("alpha_star[0..N-1]  :");
  for (Index k = 0; k < N; ++k) std::printf(" %+.4f", res.alpha_star[k]);
  std::printf("\n");

  // Sanity: equality constraint sum(alpha - alpha_star) ~ 0.
  long double sum_beta = 0.0L;
  for (Index k = 0; k < N; ++k) sum_beta += res.alpha[k] - res.alpha_star[k];
  std::printf("sum(alpha - alpha_star) = %+.3e\n", (double)sum_beta);

  return 0;
}
