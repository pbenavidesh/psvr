# Internal SMO solver for the percentage-error eps-SVR dual (Models 1 and 2).
#
# Solves, in variables u = [alpha; alpha_star] in R^{2N},
#
#   min  (1/2) beta^T Omega beta - sum_k beta_k y_k
#                                + (eps/100) sum_k (alpha_k + alpha_star_k) y_k
#   s.t. sum_k beta_k = 0;  alpha_k, alpha_star_k in [0, 100 C / y_k]
#
# where beta_k = alpha_k - alpha_star_k.  The argument `K_acc` is a kernel
# accessor (see .make_kernel_accessor) wrapping the *effective* kernel
# matrix with jitter and any (1/2) factor already applied by the caller:
# Omega for Model 1, Omega_s = (1/2)(Omega + a Omega*) for Model 2.
#
# Inside the inner loop we read kernel data via two column fetches per
# iteration -- K_p once after WSS1 picks p, K_q once after WSS3 picks q --
# and reuse them for WSS3, the 1-D step-size, and the gradient update.
# Diagonal entries come from K_acc$get_diag() (cached at construction);
# the two full matvecs (shrink rebuild and post-loop refresh) go through
# K_acc$get_matvec() so BLAS is preserved on the materialised path.  See
# the F2 design note in CLAUDE.md ("Kernel Accessor (post-F2)").
#
# The solver follows libsvm-style working-set selection on the gradient
# projection
#   tau_alpha[k]      = y_k (1 - eps/100) - (Omega beta)_k
#   tau_alphastar[k]  = y_k (1 + eps/100) - (Omega beta)_k
# choosing i by WSS1 (i = argmax tau over I_up) and j by WSS3 (the partner
# that maximises the predicted objective decrease, eq. 14 of Fan, Chen &
# Lin 2005), and stopping when (max tau over I_up) - (min tau over I_down)
# falls below `tol`.
#
# Bias recovery uses the convention f(x) = sum_k beta_k K(x_k, x) + b, so
# at the KKT optimum b = tau_v for any free support vector v.  Falls back
# to the (max I_up + min I_down)/2 sandwich when no free SV exists, which
# matches the bub_bound/blb_bound branch in the osqp path.
#
# Shrinking: every `n_check` iterations, mark variables that have stayed
# at a boundary with a "wrong-side" tau for `n_freeze` consecutive checks
# as inactive; restored from scratch (recomputing tau via
# K_acc$get_matvec(beta)) when the active-set gap drops to tol.

#' Dispatcher: SMO solver with engine choice (R reference vs Rcpp core).
#'
#' Forwards to the F7-C-full Rcpp core (`engine = "rcpp"`, default) or
#' the R reference implementation (`engine = "r"`). The R path is the
#' canonical algorithm and remains the bit-identical reference for the
#' Rcpp port; it will be deprecated in v0.0.4.0 and removed in v0.1.0
#' once the Rcpp path has cleared its graduation criteria (see CLAUDE.md
#' "engine = 'r' lifecycle").
#'
#' Warm-start projection (Algorithm 1) runs in R via `.warm_start_init()`
#' BEFORE the core call, regardless of engine, so both paths see
#' already-feasible alpha/alpha* on entry.
#'
#' @keywords internal
.smo_solve <- function(K_acc, y, C, eps,
                       tol = 1e-3, max_iter = 100000L,
                       n_check = NULL, n_freeze = 5L,
                       alpha_init = NULL,
                       alpha_star_init = NULL,
                       warm_start_check = TRUE,
                       new_mask = NULL,
                       block_k4_enabled = TRUE,
                       alpha_couple = 0.5,
                       engine = c("rcpp", "r")) {
  engine <- match.arg(engine)

  if (engine == "r") {
    return(.smo_solve_r(K_acc, y, C, eps,
                        tol = tol, max_iter = max_iter,
                        n_check = n_check, n_freeze = n_freeze,
                        alpha_init = alpha_init,
                        alpha_star_init = alpha_star_init,
                        warm_start_check = warm_start_check,
                        new_mask = new_mask,
                        block_k4_enabled = block_k4_enabled,
                        alpha_couple = alpha_couple))
  }

  # engine = "rcpp": project warm-start in R, then call core.
  alpha_init_p      <- NULL
  alpha_star_init_p <- NULL
  if (!is.null(alpha_init) || !is.null(alpha_star_init)) {
    N_y <- length(y)
    C_k <- 100 * C / y
    ws  <- .warm_start_init(alpha_init, alpha_star_init, N_y, C_k,
                            new_mask        = new_mask,
                            warm_start_check = warm_start_check)
    alpha_init_p      <- ws$alpha
    alpha_star_init_p <- ws$alpha_star
  }

  Omega <- K_acc$get_omega()
  opts <- list(
    C                 = as.numeric(C),
    eps               = as.numeric(eps),
    tol               = as.numeric(tol),
    max_iter          = as.integer(max_iter),
    n_check           = if (is.null(n_check)) -1L else as.integer(n_check),
    n_freeze          = as.integer(n_freeze),
    block_k4_enabled  = isTRUE(block_k4_enabled),
    alpha_couple      = as.numeric(alpha_couple),
    warm_start_check  = isTRUE(warm_start_check),
    alpha_init        = alpha_init_p,
    alpha_star_init   = alpha_star_init_p,
    new_mask          = new_mask
  )
  sol <- psvr_smo_fit_rcpp(Omega, y, opts)
  if (!isTRUE(sol$converged)) {
    warning(sprintf("SMO solver did not converge within max_iter = %d (final iter = %d)",
                    as.integer(max_iter), as.integer(sol$iterations)))
  }
  sol
}

#' SMO solver — R reference implementation (engine = "r").
#'
#' Canonical R-level algorithm. Bit-identical reference for the Rcpp
#' core in src/core_smo_solve.cpp. Will be deprecated in v0.0.4.0 and
#' removed in v0.1.0. Do NOT call directly; go through `.smo_solve()`.
#'
#' @keywords internal
.smo_solve_r <- function(K_acc, y, C, eps,
                       tol = 1e-3, max_iter = 100000L,
                       n_check = NULL, n_freeze = 5L,
                       alpha_init = NULL,
                       alpha_star_init = NULL,
                       warm_start_check = TRUE,
                       new_mask = NULL,
                       block_k4_enabled = TRUE,
                       alpha_couple = 0.5) {
  N     <- length(y)
  scale <- eps / 100
  C_k   <- 100 * C / y
  if (is.null(n_check)) n_check <- min(K_acc$n, 1000L)
  if (n_check < 1L)     n_check <- 1L

  # tau is in the same units as y, so apply tol relatively (per-target scale).
  # This keeps convergence behaviour consistent across kernels whose Omega
  # entries vary by orders of magnitude (e.g., RBF in [0,1] vs polynomial).
  # tol_eff is retained for the WSS3 candidate noise floor only (line ~132);
  # the convergence test uses the per-pair tol_pair scaled by max(y[p], y[k_j_w1])
  # (Theorem 8 of arXiv:2605.01446 v3).
  tol_eff <- tol * mean(y)
  y_bar   <- mean(y)

  # Theorem 3 (arXiv:2605.01446 v3): asymmetric per-sample freeze thresholds.
  # alpha-variables tied to small y_k freeze SLOWER (threshold grows like y_bar/y);
  # alpha*-variables tied to small y_k freeze FASTER (threshold floors at 1).
  # Reduces to scalar n_freeze = 5 when y is homogeneous (y_k = y_bar for all k).
  n_freeze_alpha_per <- pmax(5L, as.integer(ceiling(n_freeze * y_bar / y)))
  n_freeze_astar_per <- pmax(1L, as.integer(floor  (n_freeze * y     / y_bar)))

  diag_Omega <- K_acc$get_diag()

  # Theorem 5 (arXiv:2605.01446 v3): warm-start initialization via Algorithm 1.
  # When both alpha_init and alpha_star_init are NULL we cold-start (canonical
  # SMO entry point); otherwise we project the caller-supplied vectors to the
  # feasible region and refresh tau via one full matvec.
  if (is.null(alpha_init) && is.null(alpha_star_init)) {
    alpha         <- numeric(N)
    alpha_star    <- numeric(N)
    tau_alpha     <- y * (1 - scale)
    tau_alphastar <- y * (1 + scale)
  } else {
    ws         <- .warm_start_init(alpha_init, alpha_star_init, N, C_k,
                                   new_mask        = new_mask,
                                   warm_start_check = warm_start_check)
    alpha      <- ws$alpha
    alpha_star <- ws$alpha_star
    Kbeta      <- K_acc$get_matvec(alpha - alpha_star)
    tau_alpha     <- y * (1 - scale) - Kbeta
    tau_alphastar <- y * (1 + scale) - Kbeta
  }

  active_alpha <- rep(TRUE, N)
  active_astar <- rep(TRUE, N)
  shrink_a     <- integer(N)
  shrink_s     <- integer(N)

  tol_bound <- 1e-10
  iter      <- 0L
  converged <- FALSE

  # F7 — Block-k=4 SMO (Theorem 7 of arXiv:2605.01446 v3). When enabled,
  # each outer iteration may select a second working pair (i_2, j_2) and
  # apply a 2-D joint update; gated by the descent-guaranteed decoupling
  # criterion (D1 of the F7 spec). Pre-allocate the per-iteration log so
  # the post-loop early/late phase rates can be computed without a second
  # pass.
  joint_updates <- 0L
  k2_fallbacks  <- 0L
  joint_log     <- if (block_k4_enabled) logical(max_iter) else NULL

  while (iter < max_iter) {
    iter <- iter + 1L

    aa  <- which(active_alpha)
    ast <- which(active_astar)

    up_alpha   <- aa [alpha[aa]       < C_k[aa]  - tol_bound]
    up_astar   <- ast[alpha_star[ast] > tol_bound]
    down_alpha <- aa [alpha[aa]       > tol_bound]
    down_astar <- ast[alpha_star[ast] < C_k[ast] - tol_bound]

    if ((length(up_alpha)   + length(up_astar))   == 0L ||
        (length(down_alpha) + length(down_astar)) == 0L) break

    # ---- WSS1: i = argmax tau over I_up ----
    if (length(up_alpha) > 0L) {
      ix    <- up_alpha[which.max(tau_alpha[up_alpha])]
      tau_a <- tau_alpha[ix]
    } else { ix <- 0L; tau_a <- -Inf }
    if (length(up_astar) > 0L) {
      jx    <- up_astar[which.max(tau_alphastar[up_astar])]
      tau_s <- tau_alphastar[jx]
    } else { jx <- 0L; tau_s <- -Inf }
    if (tau_a >= tau_s) { p <- ix; i_is_alpha <- TRUE;  tau_i <- tau_a }
    else                { p <- jx; i_is_alpha <- FALSE; tau_i <- tau_s }

    # Pool I_down into a flat (idx, side, tau) list.
    low_idx_pool <- c(down_alpha, down_astar)
    low_tau_pool <- c(tau_alpha[down_alpha], tau_alphastar[down_astar])
    low_is_alpha <- c(rep(TRUE,  length(down_alpha)),
                      rep(FALSE, length(down_astar)))

    # Stopping uses the global I_down minimum (WSS1 gap).
    pos_min <- which.min(low_tau_pool)
    tau_j_w1 <- low_tau_pool[pos_min]
    Delta    <- tau_i - tau_j_w1

    # Theorem 8: per-pair tolerance, evaluated against the WSS1 convergence pair.
    # The paper text (Theorem 8 of arXiv:2605.01446 v3) says "j* = WSS3 pick", but
    # this would force WSS3 to run before the convergence test - both wasteful and
    # mathematically incorrect: Delta_w3 <= Delta_w1 by construction (WSS3 picks j
    # to maximize second-order gain, not minimize tau_j), so testing Delta_w3
    # against the tolerance would stop prematurely, before the true KKT optimality
    # gap (which equals Delta_w1) is below tolerance. The WSS1 (i_w1, j_w1) pair
    # IS the KKT optimality gap; that is the correct convergence test. This
    # deviation from literal paper text is flagged for paper-side notation fix in
    # F8 (paper TODO #4).
    k_j_w1   <- low_idx_pool[pos_min]
    tol_pair <- tol * max(y[p], y[k_j_w1])

    if (Delta <= tol_pair) {
      if (any(!active_alpha) || any(!active_astar)) {
        # Active-set converged with shrunk variables: rebuild and reactivate.
        beta_full     <- alpha - alpha_star
        Kbeta         <- K_acc$get_matvec(beta_full)
        tau_alpha     <- y * (1 - scale) - Kbeta
        tau_alphastar <- y * (1 + scale) - Kbeta
        active_alpha[] <- TRUE
        active_astar[] <- TRUE
        shrink_a[]     <- 0L
        shrink_s[]     <- 0L
        next
      }
      converged <- TRUE
      break
    }

    # ---- WSS3: pick j to maximise predicted objective decrease ----
    # Among I_down with tau_t < tau_i, score = (tau_i - tau_t)^2 / a_pt
    # where a_pt = Omega[p,p] - 2 Omega[p,t] + Omega[t,t] is the curvature
    # of the (i,j) 1-D subproblem (Fan, Chen & Lin 2005, eq. 14).
    # Column-fetch K_p once and reuse for WSS3, step-size, and the gradient
    # update.  Row-vs-column equivalence relies on the symmetry invariant
    # documented on .make_kernel_accessor().
    K_p       <- K_acc$get_column(p)
    cand_mask <- low_tau_pool < tau_i - tol_eff
    if (any(cand_mask)) {
      cand_idx <- low_idx_pool[cand_mask]
      cand_tau <- low_tau_pool[cand_mask]
      a_pt     <- K_p[p] - 2 * K_p[cand_idx] + diag_Omega[cand_idx]
      a_pt     <- pmax(a_pt, 1e-12)
      score    <- (tau_i - cand_tau)^2 / a_pt
      pos_j    <- which(cand_mask)[which.max(score)]
    } else {
      pos_j <- pos_min
    }
    q          <- low_idx_pool[pos_j]
    j_is_alpha <- low_is_alpha[pos_j]
    tau_j      <- low_tau_pool[pos_j]

    # F7: fetch K_q early so the block-k=4 xi-formula can read entries
    # from it without a second column fetch. Same call as the F4 location
    # (line ~213); just moved up. Bit-identical when block_k4_enabled = FALSE.
    K_q <- K_acc$get_column(q)

    # ---- 1-D step size (pair 1) ----
    eta       <- K_p[p] - 2 * K_p[q] + diag_Omega[q]
    R_i       <- if (i_is_alpha) C_k[p] - alpha[p] else alpha_star[p]
    R_j       <- if (j_is_alpha) alpha[q]          else C_k[q] - alpha_star[q]
    delta_max <- min(R_i, R_j)
    if (delta_max <= 0) break  # numerical safeguard
    Delta_pq  <- tau_i - tau_j
    delta     <- if (eta > 0) min(Delta_pq / eta, delta_max) else delta_max

    # ---- F7: Block-k=4 candidate selection + decoupling test ----
    # See "Phase 1 — Implementation design" in plans/frolicking-booping-forest.md
    # and Theorem 7 of arXiv:2605.01446 v3. The descent-guaranteed criterion
    # of D1 is necessary-and-sufficient (in exact arithmetic) for the joint
    # update to outperform the k=2 fallback; the literal "sufficiently small"
    # cross-coupling condition of the paper is made precise here.
    did_joint_update <- FALSE
    p2          <- NA_integer_
    q2          <- NA_integer_
    delta_2     <- 0
    K_p2        <- NULL
    i2_is_alpha <- FALSE
    j2_is_alpha <- FALSE

    if (block_k4_enabled) {
      # Step 1 — Find i_2: argmax tau over I_up \ {i_1} (sample-level
      # exclusion). i_1 = (p, i_is_alpha), so remove sample p from the
      # matching slot pool only.
      if (i_is_alpha) {
        rem_up_alpha <- up_alpha[up_alpha != p]
        if (length(rem_up_alpha) > 0L) {
          ix2_a <- rem_up_alpha[which.max(tau_alpha[rem_up_alpha])]
          tau_a2 <- tau_alpha[ix2_a]
        } else { ix2_a <- 0L; tau_a2 <- -Inf }
        if (length(up_astar) > 0L) {
          ix2_s <- up_astar[which.max(tau_alphastar[up_astar])]
          tau_s2 <- tau_alphastar[ix2_s]
        } else { ix2_s <- 0L; tau_s2 <- -Inf }
      } else {
        if (length(up_alpha) > 0L) {
          ix2_a <- up_alpha[which.max(tau_alpha[up_alpha])]
          tau_a2 <- tau_alpha[ix2_a]
        } else { ix2_a <- 0L; tau_a2 <- -Inf }
        rem_up_astar <- up_astar[up_astar != p]
        if (length(rem_up_astar) > 0L) {
          ix2_s <- rem_up_astar[which.max(tau_alphastar[rem_up_astar])]
          tau_s2 <- tau_alphastar[ix2_s]
        } else { ix2_s <- 0L; tau_s2 <- -Inf }
      }

      if (tau_a2 >= tau_s2 && tau_a2 > -Inf) {
        p2_cand <- ix2_a; i2_is_alpha_cand <- TRUE;  tau_i2 <- tau_a2
      } else if (tau_s2 > -Inf) {
        p2_cand <- ix2_s; i2_is_alpha_cand <- FALSE; tau_i2 <- tau_s2
      } else {
        p2_cand <- 0L; i2_is_alpha_cand <- FALSE; tau_i2 <- -Inf
      }

      # Sample-level disjointness from pair 1: require p2 != p and p2 != q.
      if (tau_i2 > -Inf && p2_cand != p && p2_cand != q &&
          tau_i2 > tol_eff) {
        # Step 2 — Find j_2 over I_down \ {sample p, q, p2_cand}, filtered
        # by tau_j < tau_i2 - tol_eff. WSS3 scored with decoupling
        # co-optimization (D2 of the F7 spec).
        sample_excl <- c(p, q, p2_cand)
        ca_mask <- !(down_alpha %in% sample_excl) &
                   (tau_alpha[down_alpha] < tau_i2 - tol_eff)
        cs_mask <- !(down_astar %in% sample_excl) &
                   (tau_alphastar[down_astar] < tau_i2 - tol_eff)
        cand_alpha_idx <- down_alpha[ca_mask]
        cand_astar_idx <- down_astar[cs_mask]

        if (length(cand_alpha_idx) + length(cand_astar_idx) > 0L) {
          # One extra column fetch: K_p2 needed for eta_2 and (later) the
          # fused tau update. Coupling proxy reads K_p (already in hand)
          # so no further fetch for scoring.
          K_p2 <- K_acc$get_column(p2_cand)
          diag_p <- diag_Omega[p]

          cand_idx_all <- c(cand_alpha_idx, cand_astar_idx)
          cand_tau_all <- c(tau_alpha[cand_alpha_idx],
                            tau_alphastar[cand_astar_idx])
          cand_is_alpha <- c(rep(TRUE,  length(cand_alpha_idx)),
                             rep(FALSE, length(cand_astar_idx)))

          # eta_j = Omega(p2,p2) - 2 Omega(p2, j) + Omega(j, j)
          eta_j  <- K_p2[p2_cand] - 2 * K_p2[cand_idx_all] +
                    diag_Omega[cand_idx_all]
          eta_j  <- pmax(eta_j, 1e-12)
          gain_j <- (tau_i2 - cand_tau_all)^2 / eta_j

          # Coupling proxy: |Omega(p, j)| / sqrt(Omega(p,p) * Omega(j,j)),
          # normalized correlation between pair 1's i_1-sample and the
          # candidate. Capped at 1 for Mercer kernels (Cauchy-Schwarz).
          denom <- sqrt(diag_p * diag_Omega[cand_idx_all])
          coupling_j <- ifelse(denom > 0,
                               abs(K_p[cand_idx_all]) / denom,
                               1)

          score_j_aug <- gain_j * (1 - alpha_couple * coupling_j)
          best <- which.max(score_j_aug)

          if (length(best) == 1L && score_j_aug[best] > 0) {
            q2_cand          <- cand_idx_all[best]
            tau_j2           <- cand_tau_all[best]
            j2_is_alpha_cand <- cand_is_alpha[best]

            Delta_2 <- tau_i2 - tau_j2
            eta_2   <- K_p2[p2_cand] - 2 * K_p2[q2_cand] +
                       diag_Omega[q2_cand]

            if (Delta_2 > 0 && eta_2 > 0) {
              # Step 3 — xi (corrected sign-free formula; see plan).
              # Computed from already-fetched K_p, K_q with zero extra
              # column fetches. Sign factors s_i cancel under MAPE-SVR's
              # alpha/alpha* dual because the Delta-beta update has
              # invariant +-delta sign across all 4 case branches.
              xi <- K_p[p2_cand] - K_p[q2_cand] -
                    K_q[p2_cand] + K_q[q2_cand]

              # Step 4 — Decoupling test. Delta_pq is the WSS3 gap
              # (tau_i - tau_q), the same value used at line ~198 to
              # compute pair 1's analytic delta. The descent geometry
              # of the joint update is defined by the direction pair
              # (i, q) -- pair 1's actual update direction -- not the
              # MVP convergence pair (i_w1, j_w1) from line ~136.
              lhs <- Delta_pq^2 * eta_2 + Delta_2^2 * eta
              rhs <- 2 * abs(xi * Delta_pq * Delta_2) * (1 + 1e-10)

              if (lhs > rhs) {
                # Step 5 — Compute delta_2 with pair-2 box clipping.
                R_i2 <- if (i2_is_alpha_cand) C_k[p2_cand] - alpha[p2_cand]
                        else                  alpha_star[p2_cand]
                R_j2 <- if (j2_is_alpha_cand) alpha[q2_cand]
                        else                  C_k[q2_cand] - alpha_star[q2_cand]
                delta2_max <- min(R_i2, R_j2)
                if (delta2_max > 0) {
                  did_joint_update <- TRUE
                  p2          <- p2_cand
                  q2          <- q2_cand
                  i2_is_alpha <- i2_is_alpha_cand
                  j2_is_alpha <- j2_is_alpha_cand
                  delta_2     <- min(Delta_2 / eta_2, delta2_max)
                }
              }
            }
          }
        }
      }
    }

    # ---- Apply pair 1 update (one of 4 cases) ----
    if      ( i_is_alpha &&  j_is_alpha) { alpha[p]      <- alpha[p]      + delta; alpha[q]      <- alpha[q]      - delta }
    else if ( i_is_alpha && !j_is_alpha) { alpha[p]      <- alpha[p]      + delta; alpha_star[q] <- alpha_star[q] + delta }
    else if (!i_is_alpha &&  j_is_alpha) { alpha_star[p] <- alpha_star[p] - delta; alpha[q]      <- alpha[q]      - delta }
    else                                  { alpha_star[p] <- alpha_star[p] - delta; alpha_star[q] <- alpha_star[q] + delta }

    # Numerical clip — at most 1 ulp of drift past the box.
    alpha[p]      <- max(0, min(alpha[p],      C_k[p]))
    alpha[q]      <- max(0, min(alpha[q],      C_k[q]))
    alpha_star[p] <- max(0, min(alpha_star[p], C_k[p]))
    alpha_star[q] <- max(0, min(alpha_star[q], C_k[q]))

    if (did_joint_update) {
      # ---- Apply pair 2 update (one of 4 cases, sample-disjoint from pair 1) ----
      if      ( i2_is_alpha &&  j2_is_alpha) { alpha[p2]      <- alpha[p2]      + delta_2; alpha[q2]      <- alpha[q2]      - delta_2 }
      else if ( i2_is_alpha && !j2_is_alpha) { alpha[p2]      <- alpha[p2]      + delta_2; alpha_star[q2] <- alpha_star[q2] + delta_2 }
      else if (!i2_is_alpha &&  j2_is_alpha) { alpha_star[p2] <- alpha_star[p2] - delta_2; alpha[q2]      <- alpha[q2]      - delta_2 }
      else                                    { alpha_star[p2] <- alpha_star[p2] - delta_2; alpha_star[q2] <- alpha_star[q2] + delta_2 }

      alpha[p2]      <- max(0, min(alpha[p2],      C_k[p2]))
      alpha[q2]      <- max(0, min(alpha[q2],      C_k[q2]))
      alpha_star[p2] <- max(0, min(alpha_star[p2], C_k[p2]))
      alpha_star[q2] <- max(0, min(alpha_star[q2], C_k[q2]))

      joint_updates    <- joint_updates + 1L
      joint_log[iter]  <- TRUE
    } else if (block_k4_enabled) {
      k2_fallbacks <- k2_fallbacks + 1L
      # joint_log[iter] stays FALSE
    }

    # ---- Gradient (tau) update over the active set ----
    # Pair 1 always contributes; pair 2 contributes only when joint.
    diff_pq <- K_p - K_q
    if (did_joint_update) {
      K_q2      <- K_acc$get_column(q2)
      diff_pq_2 <- K_p2 - K_q2
      if (length(aa)  > 0L) tau_alpha[aa]      <- tau_alpha[aa]      - delta * diff_pq[aa]  - delta_2 * diff_pq_2[aa]
      if (length(ast) > 0L) tau_alphastar[ast] <- tau_alphastar[ast] - delta * diff_pq[ast] - delta_2 * diff_pq_2[ast]
    } else {
      if (length(aa)  > 0L) tau_alpha[aa]      <- tau_alpha[aa]      - delta * diff_pq[aa]
      if (length(ast) > 0L) tau_alphastar[ast] <- tau_alphastar[ast] - delta * diff_pq[ast]
    }

    # ---- Shrinking (every n_check iterations) ----
    if (iter %% n_check == 0L) {
      # Use WSS1 extremes (max I_up, min I_down) for shrinking, not the WSS3
      # pair: tau_j_w1 <= tau_j and tau_i is also the WSS1 max, so this avoids
      # over-aggressive freezing of variables whose tau lies between tau_j_w1
      # and tau_j (such variables can still be informative for convergence).
      tau_i_now <- tau_i
      tau_j_now <- tau_j_w1

      cond_a_s1 <- (alpha     <= tol_bound)        & (tau_alpha     < tau_j_now)
      cond_a_s2 <- (alpha     >= C_k - tol_bound)  & (tau_alpha     > tau_i_now)
      shr_a_now <- active_alpha & (cond_a_s1 | cond_a_s2)
      shrink_a[shr_a_now]   <- shrink_a[shr_a_now] + 1L
      shrink_a[!shr_a_now]  <- 0L
      active_alpha[shrink_a >= n_freeze_alpha_per] <- FALSE   # Theorem 3: per-sample threshold

      cond_s_s3 <- (alpha_star <= tol_bound)        & (tau_alphastar > tau_i_now)
      cond_s_s4 <- (alpha_star >= C_k - tol_bound)  & (tau_alphastar < tau_j_now)
      shr_s_now <- active_astar & (cond_s_s3 | cond_s_s4)
      shrink_s[shr_s_now]   <- shrink_s[shr_s_now] + 1L
      shrink_s[!shr_s_now]  <- 0L
      active_astar[shrink_s >= n_freeze_astar_per] <- FALSE   # Theorem 3: per-sample threshold
    }
  }

  if (!converged) {
    warning(sprintf("SMO solver did not converge within max_iter = %d (final iter = %d)",
                    max_iter, iter))
  }

  # ---- Refresh full tau (post-shrinking it can be stale) ----
  beta_full     <- alpha - alpha_star
  Kbeta         <- K_acc$get_matvec(beta_full)
  tau_alpha     <- y * (1 - scale) - Kbeta
  tau_alphastar <- y * (1 + scale) - Kbeta

  # ---- Bias recovery ----
  tol_sv <- 1e-10
  free_a <- which(alpha      > tol_sv & alpha      < C_k - tol_sv)
  free_s <- which(alpha_star > tol_sv & alpha_star < C_k - tol_sv)
  tau_free <- c(tau_alpha[free_a], tau_alphastar[free_s])

  if (length(tau_free) > 0L) {
    b <- mean(tau_free)
  } else {
    # Sandwich b between WSS1 bounds, computed globally (no shrinking).
    up_a <- which(alpha      < C_k - tol_bound)
    up_s <- which(alpha_star > tol_bound)
    dn_a <- which(alpha      > tol_bound)
    dn_s <- which(alpha_star < C_k - tol_bound)
    cand_up <- c(if (length(up_a)) tau_alpha[up_a]      else NULL,
                 if (length(up_s)) tau_alphastar[up_s]  else NULL)
    cand_dn <- c(if (length(dn_a)) tau_alpha[dn_a]      else NULL,
                 if (length(dn_s)) tau_alphastar[dn_s]  else NULL)
    bub <- if (length(cand_dn)) min(cand_dn) else  Inf
    blb <- if (length(cand_up)) max(cand_up) else -Inf
    b <- if (is.finite(bub) && is.finite(blb)) (bub + blb) / 2
         else if (is.finite(bub)) bub
         else if (is.finite(blb)) blb
         else 0
  }

  # F7 telemetry: overall and early/late phase decoupling rates. The
  # early/late split tests the paper's claim that "shrinking-asymmetry
  # produces a clustered active set" -- if true, late-phase decoupling
  # rate should be substantially higher than early. Similar rates would
  # be a paper-relevant empirical finding.
  total_events <- joint_updates + k2_fallbacks
  if (block_k4_enabled && total_events > 0L) {
    decoupling_rate <- joint_updates / total_events
    iter_total      <- iter
    early_bound     <- max(50L, as.integer(ceiling(iter_total / 4)))
    late_bound      <- max(50L, as.integer(ceiling(3 * iter_total / 4)))
    early_phase_decoupling_rate <-
      if (iter_total >= 1L)
        mean(joint_log[seq_len(min(early_bound, iter_total))])
      else NA_real_
    late_phase_decoupling_rate <-
      if (iter_total >= late_bound)
        mean(joint_log[late_bound:iter_total])
      else NA_real_
  } else {
    decoupling_rate             <- NA_real_
    early_phase_decoupling_rate <- NA_real_
    late_phase_decoupling_rate  <- NA_real_
  }

  list(
    alpha                        = alpha,
    alpha_star                   = alpha_star,
    b                            = b,
    converged                    = converged,
    iterations                   = iter,
    joint_updates                = joint_updates,
    k2_fallbacks                 = k2_fallbacks,
    decoupling_rate              = decoupling_rate,
    early_phase_decoupling_rate  = early_phase_decoupling_rate,
    late_phase_decoupling_rate   = late_phase_decoupling_rate
  )
}
