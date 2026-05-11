// psvr — Portable C++ core: shared types for SMO + block-k=4 + kernels.
//
// This header is consumed by every core_*.cpp file (and by the Rcpp
// binding layer). It MUST NOT include any Rcpp types (Rcpp.h,
// NumericVector, etc.) — the core is pure C++ so a future pybind11
// binding can wrap the same code without R-isms leaking through.
//
// Memory ownership convention:
//  - All double*/Index* arguments to core functions are CALLER-OWNED.
//  - The core never allocates output buffers; it writes into pre-
//    allocated storage provided by the caller (Vec / std::vector).
//  - Vec inside FitOptions / FitResult are owning std::vectors so the
//    binding layer can copy results out into Rcpp::List.
//
// Indexing convention: Omega is stored COLUMN-MAJOR (matching R's
// default). For a sample index p ∈ [0, N), the column K_p is the
// contiguous slice Omega[p*N .. p*N + N - 1]. Omega(i, j) is at
// Omega[j*N + i].
//
// Per the F7 C-full plan, these types are stable across versions.
// The engine = "r" fallback path uses the R reference algorithm
// directly; the engine = "rcpp" path uses this core via the Rcpp
// binding layer.

#ifndef PSVR_CORE_SMO_TYPES_H
#define PSVR_CORE_SMO_TYPES_H

#include <cstddef>
#include <vector>

namespace psvr {

using Index    = std::ptrdiff_t;
using Vec      = std::vector<double>;
using IndexVec = std::vector<Index>;
// std::vector<bool> is not an addressable contiguous array; use
// unsigned char for R-interop and pointer arithmetic. 0 = false, 1 = true.
using BoolVec  = std::vector<unsigned char>;
// Used for integer-valued telemetry returned to R (IntegerVector).
using IntVec   = std::vector<int>;

// All inputs to smo_fit live here. The R-side .smo_solve() dispatcher
// fills this struct from its function arguments before calling the core.
struct FitOptions {
  double C            = 1.0;       // box parameter
  double eps          = 0.1;       // insensitivity tube half-width (percent)
  double tol          = 1e-3;      // SMO KKT-gap tolerance

  Index  max_iter     = 100000;    // outer-loop bound
  Index  n_check      = -1;        // shrinking check cadence; -1 -> min(N, 1000)
  Index  n_freeze     = 5;         // base shrinking threshold

  bool   block_k4_enabled = true;  // F7 — Theorem 7
  double alpha_couple     = 0.5;   // F7 — pair-2 coupling penalty weight

  bool   warm_start_check = false; // F5 — emit projection-residual warning

  // Already-projected warm-start state (R side runs Algorithm 1).
  // Empty vectors = cold start.
  Vec      alpha_init;
  Vec      alpha_star_init;
  BoolVec  new_mask;               // empty = infer from alpha_init == 0

  // F7.5 — record WSS1 Delta per iter into FitResult::delta_history.
  bool   trace            = false;
};

// All outputs from smo_fit live here. The R-side binding wraps this
// into an Rcpp::List with field-by-field copies.
struct FitResult {
  Vec   alpha;          // length N, pre-pruning (handed back to warm-start)
  Vec   alpha_star;     // length N
  double b              = 0.0;
  bool   converged      = false;
  Index  iterations     = 0;

  // F7 telemetry
  Index  joint_updates  = 0;
  Index  k2_fallbacks   = 0;
  double decoupling_rate              = -1.0;  // -1 sentinels for NA on R side
  double early_phase_decoupling_rate  = -1.0;
  double late_phase_decoupling_rate   = -1.0;

  // F7.5 — per-iter WSS1 Delta when FitOptions::trace; empty otherwise.
  // length == iterations on return.
  Vec    delta_history;

  // F7.6 — per-iter active-set count (sum of active_alpha + active_astar
  // masks) when FitOptions::trace; empty otherwise. length == iterations
  // on return. Used by validate_v3.R's Figure 1 active-set fraction panel.
  IntVec active_history;
};

}  // namespace psvr

#endif  // PSVR_CORE_SMO_TYPES_H
