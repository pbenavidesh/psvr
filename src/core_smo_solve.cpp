// psvr — Portable C++ core: SMO solver implementation.
//
// Direct port of R/smo_solve.R (commit 336c290:
//   feat: block-k=4 SMO inner loop (Theorem 7) in R).
// Strict bit-identicality with the R reference is the Phase 3 gate.
// Operation order, loop direction, and tie-break behavior all mirror
// R's behavior. See plans/frolicking-booping-forest.md and
// CLAUDE.md "Bit-identicality policy (post-F7)" for the discipline.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>

#include "core_smo_solve.h"
#include "core_block_k4.h"

// BLAS dgemv access. We need to match R's `Omega %*% v` exactly, which
// in R dispatches to BLAS dgemv with R's BLAS implementation. In the
// R-build, include R's BLAS headers so F77_CALL(dgemv) ... FCONE link
// to R's BLAS. In the standalone build (defining PSVR_STANDALONE_BUILD),
// declare dgemv_ via extern "C" and define no-op stubs for the macros;
// the standalone smoke check provides a stub dgemv_ symbol.
#ifdef PSVR_STANDALONE_BUILD
extern "C" {
  void dgemv_(const char* trans, const int* m, const int* n,
              const double* alpha, const double* A, const int* lda,
              const double* x, const int* incx,
              const double* beta, double* y, const int* incy);
}
#define F77_CALL(x) x ## _
#define FCONE
#else
#include <R_ext/BLAS.h>
#include <R_ext/RS.h>
#endif

namespace psvr {

namespace {

// y_out := Omega * v   (N×N dense, column-major; both inputs caller-owned)
inline void matvec(const double* Omega, Index N,
                   const double* v, double* y_out) {
  const int Ni = static_cast<int>(N);
  const int inc = 1;
  const double alpha = 1.0;
  const double beta  = 0.0;
  const char trans = 'N';
  F77_CALL(dgemv)(&trans, &Ni, &Ni, &alpha, Omega, &Ni,
                  v, &inc, &beta, y_out, &inc FCONE);
}

}  // anonymous namespace

FitResult smo_fit(const double* Omega, Index N,
                  const double* y, const FitOptions& opts) {
  FitResult res;

  // ---- Setup ----
  const double scale = opts.eps / 100.0;
  Vec C_k(N);
  for (Index k = 0; k < N; ++k) C_k[k] = 100.0 * opts.C / y[k];

  Index n_check = opts.n_check;
  if (n_check < 0) {
    n_check = (N < 1000) ? N : Index{1000};
  }
  if (n_check < 1) n_check = 1;

  // tol_eff is the per-pair tolerance reference scale: tol*mean(y).
  // R: tol_eff <- tol * mean(y)
  long double y_sum_ld = 0.0L;
  for (Index k = 0; k < N; ++k) y_sum_ld += static_cast<long double>(y[k]);
  const double y_bar   = static_cast<double>(y_sum_ld / N);
  const double tol_eff = opts.tol * y_bar;

  // F4 Theorem 3 — asymmetric per-sample freeze thresholds.
  // R:  n_freeze_alpha_per <- pmax(5L, as.integer(ceiling(n_freeze * y_bar / y)))
  //     n_freeze_astar_per <- pmax(1L, as.integer(floor  (n_freeze * y     / y_bar)))
  IndexVec n_freeze_alpha_per(N);
  IndexVec n_freeze_astar_per(N);
  for (Index k = 0; k < N; ++k) {
    const double a_raw = std::ceil(opts.n_freeze * y_bar / y[k]);
    const double s_raw = std::floor(opts.n_freeze * y[k] / y_bar);
    Index a_int = static_cast<Index>(a_raw);
    Index s_int = static_cast<Index>(s_raw);
    if (a_int < 5) a_int = 5;
    if (s_int < 1) s_int = 1;
    n_freeze_alpha_per[k] = a_int;
    n_freeze_astar_per[k] = s_int;
  }

  // Cached diagonal of Omega.
  Vec diag_Omega(N);
  for (Index k = 0; k < N; ++k) diag_Omega[k] = Omega[k * N + k];

  // Cold or warm start.
  Vec alpha(N, 0.0), alpha_star(N, 0.0);
  Vec tau_alpha(N), tau_alphastar(N);
  if (opts.alpha_init.empty() && opts.alpha_star_init.empty()) {
    for (Index k = 0; k < N; ++k) {
      tau_alpha[k]     = y[k] * (1.0 - scale);
      tau_alphastar[k] = y[k] * (1.0 + scale);
    }
  } else {
    // Caller already projected via Algorithm 1; just copy.
    if (!opts.alpha_init.empty())      alpha      = opts.alpha_init;
    if (!opts.alpha_star_init.empty()) alpha_star = opts.alpha_star_init;
    Vec beta_full(N);
    for (Index k = 0; k < N; ++k) beta_full[k] = alpha[k] - alpha_star[k];
    Vec Kbeta(N);
    matvec(Omega, N, beta_full.data(), Kbeta.data());
    for (Index k = 0; k < N; ++k) {
      tau_alpha[k]     = y[k] * (1.0 - scale) - Kbeta[k];
      tau_alphastar[k] = y[k] * (1.0 + scale) - Kbeta[k];
    }
  }

  BoolVec active_alpha(N, 1u), active_astar(N, 1u);
  IndexVec shrink_a(N, 0), shrink_s(N, 0);

  const double tol_bound = 1e-10;
  Index iter = 0;
  bool  converged = false;

  // F7 telemetry
  Index joint_updates = 0;
  Index k2_fallbacks  = 0;
  BoolVec joint_log;
  if (opts.block_k4_enabled) joint_log.assign(opts.max_iter, 0u);

  // F7.5 — per-iter WSS1 Delta when opts.trace; remains empty otherwise.
  Vec delta_history;
  if (opts.trace) delta_history.reserve(opts.max_iter);

  // Reusable scratch vectors (allocated once).
  IndexVec aa, ast;           aa.reserve(N); ast.reserve(N);
  IndexVec up_alpha, up_astar, down_alpha, down_astar;
  up_alpha.reserve(N); up_astar.reserve(N);
  down_alpha.reserve(N); down_astar.reserve(N);
  Vec       beta_buf(N);
  Vec       Kbeta_buf(N);

  // ---- Main loop ----
  while (iter < opts.max_iter) {
    ++iter;

    // active-set indices
    aa.clear(); ast.clear();
    for (Index k = 0; k < N; ++k) if (active_alpha[k]) aa.push_back(k);
    for (Index k = 0; k < N; ++k) if (active_astar[k]) ast.push_back(k);

    // up_alpha:   alpha[k]      < C_k[k] - tol_bound
    // up_astar:   alpha_star[k] > tol_bound
    // down_alpha: alpha[k]      > tol_bound
    // down_astar: alpha_star[k] < C_k[k] - tol_bound
    up_alpha.clear(); up_astar.clear();
    down_alpha.clear(); down_astar.clear();
    for (Index ix = 0; ix < (Index)aa.size(); ++ix) {
      const Index k = aa[ix];
      if (alpha[k] < C_k[k] - tol_bound) up_alpha.push_back(k);
      if (alpha[k] > tol_bound)          down_alpha.push_back(k);
    }
    for (Index ix = 0; ix < (Index)ast.size(); ++ix) {
      const Index k = ast[ix];
      if (alpha_star[k] > tol_bound)            up_astar.push_back(k);
      if (alpha_star[k] < C_k[k] - tol_bound)   down_astar.push_back(k);
    }

    if ((up_alpha.empty()   && up_astar.empty()) ||
        (down_alpha.empty() && down_astar.empty())) {
      break;
    }

    // ---- WSS1: i = argmax tau over I_up ----
    const double NEG_INF = -std::numeric_limits<double>::infinity();
    Index ix_a = -1; double tau_a = NEG_INF;
    Index ix_s = -1; double tau_s = NEG_INF;
    for (Index k = 0; k < (Index)up_alpha.size(); ++k) {
      const Index idx = up_alpha[k];
      const double v = tau_alpha[idx];
      if (ix_a < 0 || v > tau_a) { tau_a = v; ix_a = idx; }
    }
    for (Index k = 0; k < (Index)up_astar.size(); ++k) {
      const Index idx = up_astar[k];
      const double v = tau_alphastar[idx];
      if (ix_s < 0 || v > tau_s) { tau_s = v; ix_s = idx; }
    }
    Index p; bool i_is_alpha; double tau_i;
    if (tau_a >= tau_s) { p = ix_a; i_is_alpha = true;  tau_i = tau_a; }
    else                { p = ix_s; i_is_alpha = false; tau_i = tau_s; }

    // ---- Build low_* pool (alpha-side first, then alpha*-side) ----
    // We don't need to materialize the concatenation as in R; we just
    // iterate down_alpha then down_astar in that order. The pos_min /
    // tau_j_w1 / k_j_w1 logic that R implements with `c(...)` becomes
    // explicit best-tracking with the same scan order.
    Index  k_j_w1 = -1;
    double tau_j_w1 = std::numeric_limits<double>::infinity();
    for (Index k = 0; k < (Index)down_alpha.size(); ++k) {
      const Index idx = down_alpha[k];
      const double v = tau_alpha[idx];
      if (k_j_w1 < 0 || v < tau_j_w1) { tau_j_w1 = v; k_j_w1 = idx; }
    }
    for (Index k = 0; k < (Index)down_astar.size(); ++k) {
      const Index idx = down_astar[k];
      const double v = tau_alphastar[idx];
      if (k_j_w1 < 0 || v < tau_j_w1) { tau_j_w1 = v; k_j_w1 = idx; }
    }

    const double Delta = tau_i - tau_j_w1;
    if (opts.trace) delta_history.push_back(Delta);

    // F4 Theorem 8: per-pair tolerance using BOTH sample y's.
    const double tol_pair = opts.tol * std::max(y[p], y[k_j_w1]);

    if (Delta <= tol_pair) {
      bool any_inactive = false;
      for (Index k = 0; k < N; ++k) {
        if (!active_alpha[k] || !active_astar[k]) { any_inactive = true; break; }
      }
      if (any_inactive) {
        // Unshrink-rebuild: refresh tau and reactivate all variables.
        for (Index k = 0; k < N; ++k) beta_buf[k] = alpha[k] - alpha_star[k];
        matvec(Omega, N, beta_buf.data(), Kbeta_buf.data());
        for (Index k = 0; k < N; ++k) {
          tau_alpha[k]     = y[k] * (1.0 - scale) - Kbeta_buf[k];
          tau_alphastar[k] = y[k] * (1.0 + scale) - Kbeta_buf[k];
          active_alpha[k]  = 1u;
          active_astar[k]  = 1u;
          shrink_a[k]      = 0;
          shrink_s[k]      = 0;
        }
        continue;
      }
      converged = true;
      break;
    }

    // ---- WSS3: pick j to maximise predicted objective decrease ----
    const double* K_p = Omega + p * N;
    // Build candidate pool from down_alpha (alpha-side first) then
    // down_astar, filtered by tau_t < tau_i - tol_eff. R uses
    //   score = (tau_i - cand_tau)^2 / max(a_pt, 1e-12)
    // We pick argmax with first-occurrence tie-break (use strict > and
    // alpha-side-before-astar scan).
    Index  q = -1;
    bool   j_is_alpha = false;
    double best_score = -std::numeric_limits<double>::infinity();
    bool   wss3_filter_hit_any = false;
    const double tau_i_minus_tol = tau_i - tol_eff;
    for (Index k = 0; k < (Index)down_alpha.size(); ++k) {
      const Index idx = down_alpha[k];
      const double tau_t = tau_alpha[idx];
      if (!(tau_t < tau_i_minus_tol)) continue;
      wss3_filter_hit_any = true;
      double a_pt = K_p[p] - 2.0 * K_p[idx] + diag_Omega[idx];
      if (a_pt < 1e-12) a_pt = 1e-12;
      const double gap = tau_i - tau_t;
      const double score = (gap * gap) / a_pt;
      if (score > best_score) {
        best_score = score; q = idx; j_is_alpha = true;
      }
    }
    for (Index k = 0; k < (Index)down_astar.size(); ++k) {
      const Index idx = down_astar[k];
      const double tau_t = tau_alphastar[idx];
      if (!(tau_t < tau_i_minus_tol)) continue;
      wss3_filter_hit_any = true;
      double a_pt = K_p[p] - 2.0 * K_p[idx] + diag_Omega[idx];
      if (a_pt < 1e-12) a_pt = 1e-12;
      const double gap = tau_i - tau_t;
      const double score = (gap * gap) / a_pt;
      if (score > best_score) {
        best_score = score; q = idx; j_is_alpha = false;
      }
    }
    if (!wss3_filter_hit_any) {
      // R: fall back to (pos_min, low_idx_pool[pos_min]) — the WSS1 j.
      q          = k_j_w1;
      // The WSS1 j's slot: re-derive by checking which pool it was in.
      // The alpha-side wins ties (we tracked pos_min with strict <).
      // Replay: scan down_alpha for k_j_w1, if found there it's alpha.
      // If not, it's in down_astar.
      bool found_in_alpha = false;
      for (Index k = 0; k < (Index)down_alpha.size(); ++k) {
        if (down_alpha[k] == k_j_w1) { found_in_alpha = true; break; }
      }
      j_is_alpha = found_in_alpha;
    }
    const double tau_j = j_is_alpha ? tau_alpha[q] : tau_alphastar[q];

    // F7: fetch K_q early (block-k=4 xi needs it before pair-1 update).
    const double* K_q = Omega + q * N;

    // ---- 1-D step size (pair 1) ----
    const double eta = K_p[p] - 2.0 * K_p[q] + diag_Omega[q];
    const double R_i = i_is_alpha ? (C_k[p] - alpha[p]) : alpha_star[p];
    const double R_j = j_is_alpha ? alpha[q]           : (C_k[q] - alpha_star[q]);
    const double delta_max = std::min(R_i, R_j);
    if (!(delta_max > 0.0)) break;  // numerical safeguard
    const double Delta_pq = tau_i - tau_j;
    double delta;
    if (eta > 0.0) {
      const double delta_unc = Delta_pq / eta;
      delta = (delta_unc < delta_max) ? delta_unc : delta_max;
    } else {
      delta = delta_max;
    }

    // ---- F7: block-k=4 candidate selection + decoupling test ----
    BlockK4Result bk4;
    if (opts.block_k4_enabled) {
      bk4 = try_block_k4(
        N, Omega, p, q, i_is_alpha, j_is_alpha,
        eta, Delta_pq, K_p, K_q,
        diag_Omega.data(), tau_alpha.data(), tau_alphastar.data(),
        alpha.data(), alpha_star.data(), C_k.data(),
        up_alpha.data(),   (Index)up_alpha.size(),
        up_astar.data(),   (Index)up_astar.size(),
        down_alpha.data(), (Index)down_alpha.size(),
        down_astar.data(), (Index)down_astar.size(),
        tol_eff, opts.alpha_couple);
    }

    // ---- Apply pair 1 update (one of 4 cases) + clip ----
    if      ( i_is_alpha &&  j_is_alpha) { alpha[p]      += delta; alpha[q]      -= delta; }
    else if ( i_is_alpha && !j_is_alpha) { alpha[p]      += delta; alpha_star[q] += delta; }
    else if (!i_is_alpha &&  j_is_alpha) { alpha_star[p] -= delta; alpha[q]      -= delta; }
    else                                  { alpha_star[p] -= delta; alpha_star[q] += delta; }
    alpha[p]      = std::max(0.0, std::min(alpha[p],      C_k[p]));
    alpha[q]      = std::max(0.0, std::min(alpha[q],      C_k[q]));
    alpha_star[p] = std::max(0.0, std::min(alpha_star[p], C_k[p]));
    alpha_star[q] = std::max(0.0, std::min(alpha_star[q], C_k[q]));

    if (bk4.accepted) {
      // ---- Apply pair 2 update (sample-disjoint from pair 1) ----
      const Index p2 = bk4.p2;
      const Index q2 = bk4.q2;
      const double d2 = bk4.delta_2;
      if      ( bk4.i2_is_alpha &&  bk4.j2_is_alpha) { alpha[p2]      += d2; alpha[q2]      -= d2; }
      else if ( bk4.i2_is_alpha && !bk4.j2_is_alpha) { alpha[p2]      += d2; alpha_star[q2] += d2; }
      else if (!bk4.i2_is_alpha &&  bk4.j2_is_alpha) { alpha_star[p2] -= d2; alpha[q2]      -= d2; }
      else                                            { alpha_star[p2] -= d2; alpha_star[q2] += d2; }
      alpha[p2]      = std::max(0.0, std::min(alpha[p2],      C_k[p2]));
      alpha[q2]      = std::max(0.0, std::min(alpha[q2],      C_k[q2]));
      alpha_star[p2] = std::max(0.0, std::min(alpha_star[p2], C_k[p2]));
      alpha_star[q2] = std::max(0.0, std::min(alpha_star[q2], C_k[q2]));
      ++joint_updates;
      joint_log[iter - 1] = 1u;
    } else if (opts.block_k4_enabled) {
      ++k2_fallbacks;
      // joint_log[iter-1] stays 0
    }

    // ---- Gradient (tau) update over the active set ----
    // diff_pq_1 = K_p - K_q; if joint, also add delta_2 * (K_p2 - K_q2).
    const double* K_p2 = bk4.accepted ? bk4.K_p2          : nullptr;
    const double* K_q2 = bk4.accepted ? (Omega + bk4.q2 * N) : nullptr;
    if (bk4.accepted) {
      // Bit-identicality with R requires SEPARATE subtractions, not a
      // fused `tau -= (a + b)`. R writes
      //   tau_alpha[aa] <- tau_alpha[aa] - delta*diff_pq[aa] - delta_2*diff_pq_2[aa]
      // which is left-associative: (tau - a) - b. The fused form
      // `tau -= a + b` evaluates `a + b` first and rounds, giving a
      // 1-ulp drift. The dual-subtraction form matches R exactly.
      for (Index ix = 0; ix < (Index)aa.size(); ++ix) {
        const Index k = aa[ix];
        tau_alpha[k] -= delta       * (K_p[k]  - K_q[k]);
        tau_alpha[k] -= bk4.delta_2 * (K_p2[k] - K_q2[k]);
      }
      for (Index ix = 0; ix < (Index)ast.size(); ++ix) {
        const Index k = ast[ix];
        tau_alphastar[k] -= delta       * (K_p[k]  - K_q[k]);
        tau_alphastar[k] -= bk4.delta_2 * (K_p2[k] - K_q2[k]);
      }
    } else {
      for (Index ix = 0; ix < (Index)aa.size(); ++ix) {
        const Index k = aa[ix];
        tau_alpha[k]     -= delta * (K_p[k] - K_q[k]);
      }
      for (Index ix = 0; ix < (Index)ast.size(); ++ix) {
        const Index k = ast[ix];
        tau_alphastar[k] -= delta * (K_p[k] - K_q[k]);
      }
    }

    // ---- Shrinking (every n_check iterations) ----
    if (iter % n_check == 0) {
      const double tau_i_now = tau_i;
      const double tau_j_now = tau_j_w1;

      // alpha-side
      for (Index k = 0; k < N; ++k) {
        const bool cond1 = (alpha[k] <= tol_bound)        && (tau_alpha[k] < tau_j_now);
        const bool cond2 = (alpha[k] >= C_k[k] - tol_bound) && (tau_alpha[k] > tau_i_now);
        const bool shr_now = active_alpha[k] && (cond1 || cond2);
        if (shr_now) ++shrink_a[k];
        else         shrink_a[k] = 0;
        if (shrink_a[k] >= n_freeze_alpha_per[k]) active_alpha[k] = 0u;
      }
      // alpha*-side
      for (Index k = 0; k < N; ++k) {
        const bool cond3 = (alpha_star[k] <= tol_bound)        && (tau_alphastar[k] > tau_i_now);
        const bool cond4 = (alpha_star[k] >= C_k[k] - tol_bound) && (tau_alphastar[k] < tau_j_now);
        const bool shr_now = active_astar[k] && (cond3 || cond4);
        if (shr_now) ++shrink_s[k];
        else         shrink_s[k] = 0;
        if (shrink_s[k] >= n_freeze_astar_per[k]) active_astar[k] = 0u;
      }
    }
  }

  // ---- Post-loop tau refresh ----
  for (Index k = 0; k < N; ++k) beta_buf[k] = alpha[k] - alpha_star[k];
  matvec(Omega, N, beta_buf.data(), Kbeta_buf.data());
  for (Index k = 0; k < N; ++k) {
    tau_alpha[k]     = y[k] * (1.0 - scale) - Kbeta_buf[k];
    tau_alphastar[k] = y[k] * (1.0 + scale) - Kbeta_buf[k];
  }

  // ---- Bias recovery ----
  const double tol_sv = 1e-10;
  long double b_sum_ld = 0.0L;
  Index n_free = 0;
  for (Index k = 0; k < N; ++k) {
    if (alpha[k] > tol_sv && alpha[k] < C_k[k] - tol_sv) {
      b_sum_ld += static_cast<long double>(tau_alpha[k]);
      ++n_free;
    }
    if (alpha_star[k] > tol_sv && alpha_star[k] < C_k[k] - tol_sv) {
      b_sum_ld += static_cast<long double>(tau_alphastar[k]);
      ++n_free;
    }
  }
  double b = 0.0;
  if (n_free > 0) {
    b = static_cast<double>(b_sum_ld / static_cast<long double>(n_free));
  } else {
    // Sandwich path: bub = min(tau over I_down), blb = max(tau over I_up)
    double bub =  std::numeric_limits<double>::infinity();
    double blb = -std::numeric_limits<double>::infinity();
    bool has_up = false, has_dn = false;
    for (Index k = 0; k < N; ++k) {
      if (alpha[k] < C_k[k] - tol_bound) { has_up = true; if (tau_alpha[k]     > blb) blb = tau_alpha[k]; }
      if (alpha_star[k] > tol_bound)     { has_up = true; if (tau_alphastar[k] > blb) blb = tau_alphastar[k]; }
      if (alpha[k] > tol_bound)          { has_dn = true; if (tau_alpha[k]     < bub) bub = tau_alpha[k]; }
      if (alpha_star[k] < C_k[k] - tol_bound) { has_dn = true; if (tau_alphastar[k] < bub) bub = tau_alphastar[k]; }
    }
    if (has_up && has_dn)      b = (bub + blb) / 2.0;
    else if (has_dn)           b = bub;
    else if (has_up)           b = blb;
    else                       b = 0.0;
  }

  // ---- F7 telemetry: overall and early/late phase decoupling rates ----
  double decoupling_rate             = -1.0;
  double early_phase_decoupling_rate = -1.0;
  double late_phase_decoupling_rate  = -1.0;
  const Index total_events = joint_updates + k2_fallbacks;
  if (opts.block_k4_enabled && total_events > 0) {
    decoupling_rate = static_cast<double>(joint_updates)
                    / static_cast<double>(total_events);
    const Index iter_total  = iter;
    Index early_bound = static_cast<Index>(std::ceil(iter_total / 4.0));
    if (early_bound < 50) early_bound = 50;
    Index late_bound  = static_cast<Index>(std::ceil(3.0 * iter_total / 4.0));
    if (late_bound  < 50) late_bound  = 50;
    if (iter_total >= 1) {
      const Index lim = std::min<Index>(early_bound, iter_total);
      Index sum_e = 0;
      for (Index k = 0; k < lim; ++k) sum_e += static_cast<Index>(joint_log[k]);
      early_phase_decoupling_rate = static_cast<double>(sum_e) / static_cast<double>(lim);
    }
    if (iter_total >= late_bound) {
      Index sum_l = 0;
      const Index n_l = iter_total - late_bound + 1;
      for (Index k = late_bound - 1; k < iter_total; ++k) sum_l += static_cast<Index>(joint_log[k]);
      late_phase_decoupling_rate = static_cast<double>(sum_l) / static_cast<double>(n_l);
    }
  }

  // ---- Assemble FitResult ----
  res.alpha      = std::move(alpha);
  res.alpha_star = std::move(alpha_star);
  res.b          = b;
  res.converged  = converged;
  res.iterations = iter;
  res.joint_updates                = joint_updates;
  res.k2_fallbacks                 = k2_fallbacks;
  res.decoupling_rate              = decoupling_rate;
  res.early_phase_decoupling_rate  = early_phase_decoupling_rate;
  res.late_phase_decoupling_rate   = late_phase_decoupling_rate;
  res.delta_history                = std::move(delta_history);
  return res;
}

}  // namespace psvr
