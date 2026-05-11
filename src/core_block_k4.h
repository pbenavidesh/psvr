// psvr — Portable C++ core: block-k=4 SMO step (Theorem 7 of
// arXiv:2605.01446 v3).
//
// Given pair 1 (i_1, j_1) already selected by WSS1 + WSS3, try to
// select a sample-disjoint pair 2 (i_2, j_2) and apply a 2-D joint
// update via the descent-guaranteed decoupling criterion (D1 of the
// F7 design). Returns accepted=false if any gate fails; the caller
// falls back to a k=2-only update.
//
// See plans/frolicking-booping-forest.md for the F7 design context,
// the corrected sign-free xi formula derivation, and the user-resolved
// sample-level disjointness convention.

#ifndef PSVR_CORE_BLOCK_K4_H
#define PSVR_CORE_BLOCK_K4_H

#include "core_smo_types.h"

namespace psvr {

struct BlockK4Result {
  bool   accepted     = false;  // true => apply 2-D joint update; false => k=2 fallback
  Index  p2           = -1;     // sample index for i_2
  Index  q2           = -1;     // sample index for j_2
  bool   i2_is_alpha  = false;  // slot for i_2 (true = α-slot, false = α*-slot)
  bool   j2_is_alpha  = false;  // slot for j_2
  double delta_2      = 0.0;    // analytic 1-D step for pair 2 (already box-clipped)
  // K_p2 (column of Omega for sample p2) is needed by the caller for
  // the fused tau update. Caller can recompute as Omega + p2 * N to
  // avoid a returned pointer; we expose it here purely for clarity.
  const double* K_p2 = nullptr;
};

// `up_alpha`, `up_astar`, `down_alpha`, `down_astar` are 0-based sample
// index arrays of length n_*. These are the active-set masks built by
// smo_fit in its outer loop. The block-k=4 helper does not need the
// alpha boolean masks (active_alpha / active_astar) because the up/down
// arrays already filter to active samples.
BlockK4Result try_block_k4(
    Index N,
    const double* Omega,          // column-major N×N
    Index p, Index q,
    bool i_is_alpha, bool j_is_alpha,
    double eta_1, double Delta_pq,
    const double* K_p,            // = Omega + p * N
    const double* K_q,            // = Omega + q * N
    const double* diag_Omega,     // length N, pre-computed diagonal
    const double* tau_alpha,      // length N
    const double* tau_alphastar,  // length N
    const double* alpha,          // length N
    const double* alpha_star,     // length N
    const double* C_k,            // length N, per-sample box bound
    const Index*  up_alpha,    Index n_up_alpha,
    const Index*  up_astar,    Index n_up_astar,
    const Index*  down_alpha,  Index n_down_alpha,
    const Index*  down_astar,  Index n_down_astar,
    double tol_eff,
    double alpha_couple);

}  // namespace psvr

#endif  // PSVR_CORE_BLOCK_K4_H
