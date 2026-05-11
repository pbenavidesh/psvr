// psvr — Portable C++ core: SMO solver for the percentage-error
// epsilon-SVR dual (Models 1 and 2).
//
// Pure-C++ port of R/smo_solve.R. The R-side dispatcher (.smo_solve
// with engine = "rcpp") feeds an already-jittered Omega (Model 1) or
// Omega_s (Model 2 — with adaptive spectral shift already applied)
// into smo_fit. The warm-start projection (Algorithm 1) runs in R
// before the call; the core receives already-feasible alpha_init /
// alpha_star_init in FitOptions.
//
// Bit-identicality with the R reference (engine = "r") is the strict
// Phase 3 gate. The implementation matches R's:
//   - loop directions and traversal orders
//   - long-double accumulator pattern (where applicable)
//   - which.max first-occurrence-of-max tie-breaking
//   - alpha-pool-before-astar-pool concatenation order for I_down
//   - BLAS dgemv for the matvec (warm-start refresh + post-loop
//     refresh + unshrink-rebuild), invoked via F77_CALL(dgemv) +
//     FCONE so R's BLAS implementation is used directly.

#ifndef PSVR_CORE_SMO_SOLVE_H
#define PSVR_CORE_SMO_SOLVE_H

#include "core_smo_types.h"

namespace psvr {

// Solve the MAPE epsilon-SVR dual on the given precomputed (and
// jittered / shifted) Omega. The caller owns Omega and y; the return
// value owns its FitResult vectors via std::vector.
FitResult smo_fit(const double* Omega,   // N×N, column-major
                  Index N,
                  const double* y,
                  const FitOptions& opts);

}  // namespace psvr

#endif  // PSVR_CORE_SMO_SOLVE_H
