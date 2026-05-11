// psvr — Portable C++ core: block-k=4 SMO step implementation.
//
// Direct port of the R block-k=4 logic in R/smo_solve.R (committed as
// part of feat: block-k=4 SMO inner loop (Theorem 7) in R). Strict
// bit-identicality with the R reference is the Phase 3 gate. Loop
// directions and tie-break behavior match R's which.max (first-
// occurrence-of-max) by construction (std::max-tracking with > rather
// than >=).

#include <algorithm>
#include <cmath>
#include <limits>
#include "core_block_k4.h"

namespace psvr {

namespace {

// Replicate R's which.max: returns index of the FIRST occurrence of
// the maximum value among `values[indices[0..n)]`. Returns -1 on empty
// input (caller treats as "no candidate"). Tie-break: first index in
// scan order.
inline Index argmax_indirect(const double* values,
                             const Index* indices, Index n) {
  if (n <= 0) return -1;
  Index best = indices[0];
  double best_val = values[best];
  for (Index k = 1; k < n; ++k) {
    const Index idx = indices[k];
    const double v = values[idx];
    if (v > best_val) {
      best_val = v;
      best     = idx;
    }
  }
  return best;
}

// Same, but skipping a single excluded sample index `excl`.
// Returns -1 if no element survives the exclusion.
inline Index argmax_indirect_excl(const double* values,
                                  const Index* indices, Index n,
                                  Index excl) {
  Index   best     = -1;
  double  best_val = -std::numeric_limits<double>::infinity();
  for (Index k = 0; k < n; ++k) {
    const Index idx = indices[k];
    if (idx == excl) continue;
    const double v = values[idx];
    if (best < 0 || v > best_val) {
      best_val = v;
      best     = idx;
    }
  }
  return best;
}

}  // anonymous namespace

BlockK4Result try_block_k4(
    Index N,
    const double* Omega,
    Index p, Index q,
    bool i_is_alpha, bool /*j_is_alpha*/,
    double eta_1, double Delta_pq,
    const double* K_p, const double* K_q,
    const double* diag_Omega,
    const double* tau_alpha, const double* tau_alphastar,
    const double* alpha,     const double* alpha_star,
    const double* C_k,
    const Index* up_alpha,    Index n_up_alpha,
    const Index* up_astar,    Index n_up_astar,
    const Index* down_alpha,  Index n_down_alpha,
    const Index* down_astar,  Index n_down_astar,
    double tol_eff,
    double alpha_couple)
{
  BlockK4Result res;
  const double NEG_INF = -std::numeric_limits<double>::infinity();

  // ---- Step 1: find i_2 = argmax tau over I_up \ {i_1} ----
  // i_1 occupies (p, slot=i_is_alpha). Exclude p from the matching slot pool only.
  Index  ix2_a   = -1;
  double tau_a2  = NEG_INF;
  Index  ix2_s   = -1;
  double tau_s2  = NEG_INF;
  if (i_is_alpha) {
    ix2_a = argmax_indirect_excl(tau_alpha,     up_alpha, n_up_alpha, p);
    if (ix2_a >= 0) tau_a2 = tau_alpha[ix2_a];
    ix2_s = argmax_indirect(tau_alphastar, up_astar, n_up_astar);
    if (ix2_s >= 0) tau_s2 = tau_alphastar[ix2_s];
  } else {
    ix2_a = argmax_indirect(tau_alpha, up_alpha, n_up_alpha);
    if (ix2_a >= 0) tau_a2 = tau_alpha[ix2_a];
    ix2_s = argmax_indirect_excl(tau_alphastar, up_astar, n_up_astar, p);
    if (ix2_s >= 0) tau_s2 = tau_alphastar[ix2_s];
  }

  Index  p2_cand = -1;
  bool   i2_is_alpha_cand = false;
  double tau_i2  = NEG_INF;
  // R: `if (tau_a2 >= tau_s2 && tau_a2 > -Inf)`. tau_a2 is -Inf when
  // ix2_a < 0, so `tau_a2 > -Inf` covers the empty-pool case.
  if (tau_a2 >= tau_s2 && tau_a2 > NEG_INF) {
    p2_cand          = ix2_a;
    i2_is_alpha_cand = true;
    tau_i2           = tau_a2;
  } else if (tau_s2 > NEG_INF) {
    p2_cand          = ix2_s;
    i2_is_alpha_cand = false;
    tau_i2           = tau_s2;
  } else {
    return res;  // no candidate at all
  }

  // Sample-level disjointness with pair 1; noise-floor on tau_i2.
  if (p2_cand == p || p2_cand == q) return res;
  if (!(tau_i2 > tol_eff))           return res;

  // K_p2 column needed for eta_2 and the (later) fused tau update.
  const double* K_p2 = Omega + p2_cand * N;

  // ---- Step 2: find j_2 via WSS3 with decoupling co-optimization ----
  // Score: gain_j * (1 - alpha_couple * coupling_j) where
  //   eta_j      = Ω(p2,p2) - 2 Ω(p2,j) + Ω(j,j)
  //   gain_j     = (tau_i2 - tau_j)^2 / max(eta_j, 1e-12)
  //   coupling_j = |Ω(p, j)| / sqrt(Ω(p,p) * Ω(j,j))   (1 if denom <= 0)
  //
  // R uses two pools (alpha-side and alpha*-side) concatenated; the
  // alpha-side comes first. We preserve this scan order so any tied
  // score breaks to the same candidate R picks (first-in-scan).

  const double diag_p           = diag_Omega[p];
  const double diag_p2          = K_p2[p2_cand];        // = Omega(p2, p2)
  const double tau_i2_minus_tol = tau_i2 - tol_eff;

  double best_score    = -std::numeric_limits<double>::infinity();
  Index  best_q2       = -1;
  bool   best_q2_alpha = false;

  auto scan_pool = [&](const Index* pool, Index n_pool,
                       const double* tau_pool, bool slot_is_alpha) {
    for (Index k = 0; k < n_pool; ++k) {
      const Index j = pool[k];
      // sample-level disjointness from {p, q, p2_cand}
      if (j == p || j == q || j == p2_cand) continue;
      const double tau_j = tau_pool[j];
      if (!(tau_j < tau_i2_minus_tol)) continue;

      // eta_j with pmax(., 1e-12)
      double eta_j = diag_p2 - 2.0 * K_p2[j] + diag_Omega[j];
      if (eta_j < 1e-12) eta_j = 1e-12;

      const double gap = tau_i2 - tau_j;
      const double gain = (gap * gap) / eta_j;

      // coupling_j
      const double denom2 = diag_p * diag_Omega[j];
      double coupling;
      if (denom2 > 0.0) {
        const double abs_kpj = std::fabs(K_p[j]);
        coupling = abs_kpj / std::sqrt(denom2);
      } else {
        coupling = 1.0;  // max penalty
      }

      const double score = gain * (1.0 - alpha_couple * coupling);

      // R's which.max picks FIRST occurrence of max; we use strict >
      // to keep that semantic (alpha-side scanned first, so equal
      // scores in the astar pool do not displace alpha-side picks).
      if (score > best_score) {
        best_score    = score;
        best_q2       = j;
        best_q2_alpha = slot_is_alpha;
      }
    }
  };

  scan_pool(down_alpha, n_down_alpha, tau_alpha,     /*slot_is_alpha=*/true);
  scan_pool(down_astar, n_down_astar, tau_alphastar, /*slot_is_alpha=*/false);

  if (best_q2 < 0)        return res;  // empty pool
  if (!(best_score > 0))  return res;  // all candidates suppressed by coupling

  // ---- Step 3: compute Δ_2, η_2, ξ ----
  const double tau_j2 = best_q2_alpha ? tau_alpha[best_q2] : tau_alphastar[best_q2];
  const double Delta_2 = tau_i2 - tau_j2;
  if (!(Delta_2 > 0.0)) return res;

  const double eta_2 = diag_p2 - 2.0 * K_p2[best_q2] + diag_Omega[best_q2];
  if (!(eta_2 > 0.0)) return res;

  // xi via the corrected sign-free formula (sign factors cancel under
  // MAPE-SVR's α/α* dual).
  const double xi = K_p[p2_cand] - K_p[best_q2] - K_q[p2_cand] + K_q[best_q2];

  // ---- Step 4: decoupling test (necessary-and-sufficient) ----
  // Use Delta_pq (WSS3 gap, same value as pair 1's delta_1 numerator),
  // NOT the WSS1 convergence gap.
  const double lhs = Delta_pq * Delta_pq * eta_2 + Delta_2 * Delta_2 * eta_1;
  const double rhs = 2.0 * std::fabs(xi * Delta_pq * Delta_2) * (1.0 + 1e-10);
  if (!(lhs > rhs)) return res;

  // ---- Step 5: compute delta_2 with box clipping ----
  const double R_i2 = i2_is_alpha_cand
                        ? (C_k[p2_cand]   - alpha[p2_cand])
                        : (alpha_star[p2_cand]);
  const double R_j2 = best_q2_alpha
                        ? (alpha[best_q2])
                        : (C_k[best_q2] - alpha_star[best_q2]);
  const double delta2_max = std::min(R_i2, R_j2);
  if (!(delta2_max > 0.0)) return res;

  const double delta_unc = Delta_2 / eta_2;
  const double delta_2   = (delta_unc < delta2_max) ? delta_unc : delta2_max;

  // All gates passed — joint update viable.
  res.accepted    = true;
  res.p2          = p2_cand;
  res.q2          = best_q2;
  res.i2_is_alpha = i2_is_alpha_cand;
  res.j2_is_alpha = best_q2_alpha;
  res.delta_2     = delta_2;
  res.K_p2        = K_p2;
  return res;
}

}  // namespace psvr
