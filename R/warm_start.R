# Warm-start initialization for the percentage-error eps-SVR dual.
#
# Implements Algorithm 1 of arXiv:2605.01446 v3 (Theorem 5).  Callers supply
# candidate (alpha, alpha_star) vectors of length N -- typically from a
# previous converged fit, with samples not in the new fold zero-filled -- and
# this helper projects them onto the feasible region:
#
#   Step 1 (caller's responsibility): zero-fill samples not in S_prev ∩ S_new.
#   Step 2: project onto the equality constraint sum(alpha - alpha_star) = 0
#           by distributing the violation across NEW samples only (see below).
#   Step 3: clip each variable into its per-sample box [0, C_k].
#   Step 4 (optional): assert feasibility of the projected pair.
#
# Paper deviation (F5 paper TODO #6): the published Algorithm 1 Step 2
# applies a uniform shift across ALL N samples, including retained samples
# from the previous fold's converged solution.  We diverge: the violation
# arises ENTIRELY from removed samples (S_prev \ S_new) whose alpha values
# are no longer used, and retained samples (S_prev ∩ S_new) were at the
# equality-constraint manifold at the previous fold's optimum.  Distributing
# uniformly perturbs retained samples and degrades the warm-start
# information; distributing over NEW samples only (S_new \ S_prev) preserves
# the converged values exactly and absorbs the entire violation in samples
# that had no prior information.  Both options leave post-Step-2 violation =
# 0; the Glasmachers-Igel (2006) convergence proof applies to any feasible
# initialisation.  Empirically the new-samples-only variant rescues T5's
# predicted CV speedup on heterogeneous fixtures.  smo-v3.tex Algorithm 1
# Step 2 will be updated accordingly in F8 (paper TODO #6).

.warm_start_init <- function(alpha_init, alpha_star_init, N, C_k,
                             new_mask = NULL,
                             warm_start_check = TRUE) {
  if (is.null(alpha_init))      alpha_init      <- numeric(N)
  if (is.null(alpha_star_init)) alpha_star_init <- numeric(N)

  if (length(alpha_init) != N)
    stop(sprintf("`alpha_init` has length %d but training set has N = %d.",
                 length(alpha_init), N))
  if (length(alpha_star_init) != N)
    stop(sprintf("`alpha_star_init` has length %d but training set has N = %d.",
                 length(alpha_star_init), N))

  alpha      <- as.numeric(alpha_init)
  alpha_star <- as.numeric(alpha_star_init)

  # Resolve new_mask: explicit logical vector, or infer from "both sides
  # exactly zero" (the cold-start signal for samples with no prior info).
  if (is.null(new_mask)) {
    new_mask <- (alpha == 0) & (alpha_star == 0)
  } else {
    if (!is.logical(new_mask) || length(new_mask) != N)
      stop(sprintf("`new_mask` must be a logical vector of length %d.", N))
  }

  # Step 2: equality projection sum(alpha - alpha_star) = 0.
  # Distribute violation over NEW samples only; fall back to paper-style
  # uniform shift if there are no new samples (e.g., same training set
  # as the previous fold).
  violation <- sum(alpha - alpha_star)
  n_new     <- sum(new_mask)
  if (n_new > 0L) {
    per_new       <- violation / n_new
    alpha[new_mask] <- alpha[new_mask] - per_new
  } else {
    shift <- violation / N
    alpha <- alpha - shift
  }

  # Step 3: per-sample box [0, C_k].
  alpha      <- pmin(pmax(alpha,      0), C_k)
  alpha_star <- pmin(pmax(alpha_star, 0), C_k)

  # Step 3b: a single one-pass refinement when box-clipping leaves a residual
  # equality violation (e.g. when the per-new shift forces clipping at 0 or
  # C_k for some new samples).  Distribute the residual uniformly over all
  # N samples as a safety net.
  resid <- sum(alpha - alpha_star)
  if (abs(resid) > 1e-10 * max(C_k)) {
    alpha      <- pmin(pmax(alpha      - resid / N, 0), C_k)
  }

  if (isTRUE(warm_start_check)) {
    eq_resid  <- abs(sum(alpha - alpha_star))
    eq_tol    <- 1e-6 * max(C_k)
    if (eq_resid > eq_tol) {
      warning(sprintf(
        paste0("warm-start equality residual %.2e exceeds tolerance %.2e ",
               "after projection. SMO will still converge but the speedup ",
               "is reduced."),
        eq_resid, eq_tol))
    }
    if (any(alpha      < 0 - 1e-12) || any(alpha      > C_k + 1e-12))
      stop("warm-start alpha violates per-sample box after projection.")
    if (any(alpha_star < 0 - 1e-12) || any(alpha_star > C_k + 1e-12))
      stop("warm-start alpha_star violates per-sample box after projection.")
  }

  list(alpha = alpha, alpha_star = alpha_star)
}
