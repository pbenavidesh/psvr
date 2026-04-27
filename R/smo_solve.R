# Internal SMO solver for the percentage-error eps-SVR dual (Models 1 and 2).
#
# Solves, in variables u = [alpha; alpha_star] in R^{2N},
#
#   min  (1/2) beta^T Omega beta - sum_k beta_k y_k
#                                + (eps/100) sum_k (alpha_k + alpha_star_k) y_k
#   s.t. sum_k beta_k = 0;  alpha_k, alpha_star_k in [0, 100 C / y_k]
#
# where beta_k = alpha_k - alpha_star_k.  The argument `Omega` is the
# *effective* kernel matrix (jitter and any (1/2) factor already applied):
# Omega for Model 1, 0.5 * Ks for Model 2.
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
# as inactive; restored from scratch (recomputing tau via Omega %*% beta)
# when the active-set gap drops to tol.

.smo_solve <- function(Omega, y, C, eps,
                       tol = 1e-3, max_iter = 100000L,
                       n_check = NULL, n_freeze = 5L) {
  N     <- length(y)
  scale <- eps / 100
  C_k   <- 100 * C / y
  if (is.null(n_check)) n_check <- min(N, 1000L)
  if (n_check < 1L)     n_check <- 1L

  # tau is in the same units as y, so apply tol relatively (per-target scale).
  # This keeps convergence behaviour consistent across kernels whose Omega
  # entries vary by orders of magnitude (e.g., RBF in [0,1] vs polynomial).
  tol_eff <- tol * mean(y)

  diag_Omega    <- diag(Omega)
  alpha         <- numeric(N)
  alpha_star    <- numeric(N)
  tau_alpha     <- y * (1 - scale)
  tau_alphastar <- y * (1 + scale)

  active_alpha <- rep(TRUE, N)
  active_astar <- rep(TRUE, N)
  shrink_a     <- integer(N)
  shrink_s     <- integer(N)

  tol_bound <- 1e-10
  iter      <- 0L
  converged <- FALSE

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

    if (Delta <= tol_eff) {
      if (any(!active_alpha) || any(!active_astar)) {
        # Active-set converged with shrunk variables: rebuild and reactivate.
        beta_full     <- alpha - alpha_star
        Kbeta         <- as.numeric(Omega %*% beta_full)
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
    cand_mask <- low_tau_pool < tau_i - tol_eff
    if (any(cand_mask)) {
      cand_idx <- low_idx_pool[cand_mask]
      cand_tau <- low_tau_pool[cand_mask]
      a_pt     <- Omega[p, p] - 2 * Omega[p, cand_idx] + diag_Omega[cand_idx]
      a_pt     <- pmax(a_pt, 1e-12)
      score    <- (tau_i - cand_tau)^2 / a_pt
      pos_j    <- which(cand_mask)[which.max(score)]
    } else {
      pos_j <- pos_min
    }
    q          <- low_idx_pool[pos_j]
    j_is_alpha <- low_is_alpha[pos_j]
    tau_j      <- low_tau_pool[pos_j]

    # ---- 1-D step size ----
    eta       <- Omega[p, p] - 2 * Omega[p, q] + diag_Omega[q]
    R_i       <- if (i_is_alpha) C_k[p] - alpha[p] else alpha_star[p]
    R_j       <- if (j_is_alpha) alpha[q]          else C_k[q] - alpha_star[q]
    delta_max <- min(R_i, R_j)
    if (delta_max <= 0) break  # numerical safeguard
    Delta_pq  <- tau_i - tau_j
    delta     <- if (eta > 0) min(Delta_pq / eta, delta_max) else delta_max

    # ---- Apply update (one of 4 cases) ----
    if      ( i_is_alpha &&  j_is_alpha) { alpha[p]      <- alpha[p]      + delta; alpha[q]      <- alpha[q]      - delta }
    else if ( i_is_alpha && !j_is_alpha) { alpha[p]      <- alpha[p]      + delta; alpha_star[q] <- alpha_star[q] + delta }
    else if (!i_is_alpha &&  j_is_alpha) { alpha_star[p] <- alpha_star[p] - delta; alpha[q]      <- alpha[q]      - delta }
    else                                  { alpha_star[p] <- alpha_star[p] - delta; alpha_star[q] <- alpha_star[q] + delta }

    # Numerical clip — at most 1 ulp of drift past the box.
    alpha[p]      <- max(0, min(alpha[p],      C_k[p]))
    alpha[q]      <- max(0, min(alpha[q],      C_k[q]))
    alpha_star[p] <- max(0, min(alpha_star[p], C_k[p]))
    alpha_star[q] <- max(0, min(alpha_star[q], C_k[q]))

    # ---- Gradient (tau) update over the active set ----
    diff_pq <- Omega[, p] - Omega[, q]
    if (length(aa)  > 0L) tau_alpha[aa]      <- tau_alpha[aa]      - delta * diff_pq[aa]
    if (length(ast) > 0L) tau_alphastar[ast] <- tau_alphastar[ast] - delta * diff_pq[ast]

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
      active_alpha[shrink_a >= n_freeze] <- FALSE

      cond_s_s3 <- (alpha_star <= tol_bound)        & (tau_alphastar > tau_i_now)
      cond_s_s4 <- (alpha_star >= C_k - tol_bound)  & (tau_alphastar < tau_j_now)
      shr_s_now <- active_astar & (cond_s_s3 | cond_s_s4)
      shrink_s[shr_s_now]   <- shrink_s[shr_s_now] + 1L
      shrink_s[!shr_s_now]  <- 0L
      active_astar[shrink_s >= n_freeze] <- FALSE
    }
  }

  if (!converged) {
    warning(sprintf("SMO solver did not converge within max_iter = %d (final iter = %d)",
                    max_iter, iter))
  }

  # ---- Refresh full tau (post-shrinking it can be stale) ----
  beta_full     <- alpha - alpha_star
  Kbeta         <- as.numeric(Omega %*% beta_full)
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

  list(
    alpha      = alpha,
    alpha_star = alpha_star,
    b          = b,
    converged  = converged,
    iterations = iter
  )
}
