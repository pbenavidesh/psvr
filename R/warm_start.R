# Warm-start initialization for the percentage-error eps-SVR dual.
#
# Implements Step 2-4 of Algorithm 1 of arXiv:2605.01446 v3 (Theorem 5).
# Callers supply candidate (alpha, alpha_star) vectors of length N --
# typically from a previous converged fit, with samples not in the new fold
# zero-filled -- and this helper projects them onto the feasible region:
#
#   Step 1 (caller's responsibility): zero-fill samples not in S_prev n S_new.
#   Step 2-3: EXACT Euclidean projection onto
#               {(alpha, alpha*) : sum(alpha - alpha*) = 0, 0 <= . <= C_k}
#             via a single scalar multiplier (waterfilling).
#   Step 4 (optional): assert feasibility of the projected pair.
#
# ---------------------------------------------------------------------------
# Why an exact projection (supersedes the pre-0.0.2.9010 shift-then-clip)
# ---------------------------------------------------------------------------
# The previous implementation applied a uniform shift to `alpha` on the NEW
# samples only, then clipped into the box.  New samples are new precisely
# because they carry no prior information, so alpha_k = 0 there by
# construction.  Whenever the equality violation was POSITIVE the shift drove
# those entries negative and the box clip returned every one of them to 0 --
# destroying 100% of the correction, deterministically, not occasionally.
#
# That was not merely a lost speedup.  SMO's pairwise updates conserve
# sum(alpha - alpha*) exactly, so an infeasible starting point is never
# repaired: the solver converges on the wrong affine manifold and returns the
# optimum of a different problem.  Measured on a 5-fold fixture, the surviving
# residual reached 5.8e+02 and warm-start predictions diverged from cold-start
# by >1000% of mean|y|.
#
# The fix computes the true minimum-norm projection.  The KKT conditions of
#
#     min ||x - x0||^2   s.t.   a'x = 0,  0 <= x <= C     (x = (alpha, alpha*),
#                                                          a = (+1.., -1..))
#
# give x_k(lambda) = clip(x0_k - lambda * a_k, 0, C_k) for a single scalar
# multiplier lambda on the equality row, i.e.
#
#     alpha_k(lambda)  = clip(alpha_k  - lambda, 0, C_k)
#     alpha*_k(lambda) = clip(alpha*_k + lambda, 0, C_k)
#
# and lambda is fixed by the scalar root-find g(lambda) = 0, where
# g(lambda) = sum(alpha(lambda)) - sum(alpha*(lambda)).
#
# MONOTONICITY.  clip(u, 0, C) is continuous and non-decreasing in u.  In
# g, every alpha term composes it with (alpha_k - lambda), non-increasing in
# lambda; every alpha* term composes it with (alpha*_k + lambda),
# non-decreasing in lambda, and enters negated.  So g is a sum of continuous
# non-increasing functions: continuous, non-increasing, hence its root set is
# a non-empty closed interval once a sign change is bracketed, and any root
# yields the same projected point.
#
# BRACKET.  At lambda_hi = max(0, max(alpha0)) every (alpha0_k - lambda_hi)
# is <= 0, so all alpha(lambda_hi) = 0 and g(lambda_hi) = -sum(alpha*) <= 0.
# At lambda_lo = -max(0, max(alpha*0)) every (alpha*0_k + lambda_lo) is <= 0,
# so all alpha*(lambda_lo) = 0 and g(lambda_lo) = sum(alpha) >= 0.  Also
# lambda_lo <= 0 <= lambda_hi.  Hence [lambda_lo, lambda_hi] brackets a root
# unconditionally -- including when x0 lies outside the box, since the clip
# in the formula is what enforces the box.  When x0 IS in the box (the
# psvr_cv() path always is), g(0) equals the raw equality violation.
#
# ---------------------------------------------------------------------------
# Scope of the projection: ALL variables, not new samples only
# ---------------------------------------------------------------------------
# Pre-0.0.2.9010 the shift ranged over S_new \ S_prev only, on the paper
# TODO #6 rationale that this preserves retained samples' converged values
# exactly.  The exact projection ranges over ALL 2N variables and will perturb
# retained values slightly.  Two reasons this is the right trade:
#
#   * Minimum-norm IS what warm-starting wants: the closest feasible point to
#     the previous optimum.  Retained values were optimal for the PREVIOUS
#     fold's problem, not this one, so exact preservation has no claim on
#     correctness -- only proximity does, which is what the norm measures.
#   * Restricting to new samples is not always feasible.  With retained values
#     held fixed the new block must absorb the whole violation, which requires
#     |violation| <= sum(C_k) over new samples; that can fail outright, and
#     fails degenerately whenever n_new = 0 (identical training set across
#     folds), where the reachable set collapses to {0}.  The all-variable
#     feasible set, by contrast, is never empty: x = 0 always satisfies both
#     the equality and the box.
#
# The `new_mask` argument is retained for API compatibility (it is part of the
# psvr() / .smo_solve() signatures and is passed by psvr_cv()) but no longer
# influences the projection.

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
  if (!is.null(new_mask)) {
    if (!is.logical(new_mask) || length(new_mask) != N)
      stop(sprintf("`new_mask` must be a logical vector of length %d.", N))
  }

  alpha0      <- as.numeric(alpha_init)
  alpha_star0 <- as.numeric(alpha_star_init)

  if (N == 0L)
    return(list(alpha = alpha0, alpha_star = alpha_star0))

  # g(lambda) = sum(alpha(lambda)) - sum(alpha*(lambda)); see header for the
  # monotonicity argument and the bracket derivation.
  g <- function(lambda) {
    sum(pmin(pmax(alpha0      - lambda, 0), C_k)) -
    sum(pmin(pmax(alpha_star0 + lambda, 0), C_k))
  }

  lo <- -max(0, max(alpha_star0))   # g(lo) >= 0
  hi <-  max(0, max(alpha0))        # g(hi) <= 0

  if (g(lo) <= 0) {
    lambda <- lo
  } else if (g(hi) >= 0) {
    lambda <- hi
  } else {
    # Bisection preserving g(lo) >= 0 >= g(hi).  Chosen over a sorting-based
    # exact breakpoint search because the invariant is checkable by eye and
    # the cost is irrelevant here (one projection per fold, O(N) per step);
    # the closed-form polish below recovers the exactness the breakpoint
    # search would have given directly.
    for (it in seq_len(200L)) {
      mid <- 0.5 * (lo + hi)
      if (mid <= lo || mid >= hi) break   # bracket collapsed to adjacent doubles
      if (g(mid) > 0) lo <- mid else hi <- mid
    }
    lambda <- 0.5 * (lo + hi)
  }

  # Closed-form polish.  Bisection pins lambda to ~1 ulp, but g has slope up
  # to -(2N), so g(lambda) can still sit around 1e-10 in absolute terms.  With
  # the active set frozen at `lambda`, g is affine and its root is exact:
  #
  #   lambda* = [ sum_{free a} alpha0 + sum_{sat a} C_k
  #             - sum_{free a*} alpha*0 - sum_{sat a*} C_k ] / n_free
  #
  # Accepted only if it reproduces the same active set, so a breakpoint
  # straddle falls back to the bisection value rather than overshooting.
  a_arg  <- alpha0      - lambda
  s_arg  <- alpha_star0 + lambda
  free_a <- a_arg > 0 & a_arg < C_k
  free_s <- s_arg > 0 & s_arg < C_k
  n_free <- sum(free_a) + sum(free_s)
  if (n_free > 0L) {
    lambda_p <- ((sum(alpha0[free_a])      + sum(C_k[a_arg >= C_k])) -
                 (sum(alpha_star0[free_s]) + sum(C_k[s_arg >= C_k]))) / n_free
    ap <- alpha0 - lambda_p
    sp <- alpha_star0 + lambda_p
    if (identical(ap > 0 & ap < C_k, free_a) &&
        identical(sp > 0 & sp < C_k, free_s))
      lambda <- lambda_p
  }

  alpha      <- pmin(pmax(alpha0      - lambda, 0), C_k)
  alpha_star <- pmin(pmax(alpha_star0 + lambda, 0), C_k)

  if (isTRUE(warm_start_check)) {
    # Escalated from warning() to stop() at 0.0.2.9010.  SMO conserves
    # sum(alpha - alpha*) exactly, so a surviving equality residual is not a
    # degraded speedup -- it guarantees the solver returns the optimum of a
    # different problem.  This now matches how box violations are treated
    # below.  Callers that genuinely want an unprojected start opt out with
    # warm_start_check = FALSE.
    #
    # Tolerance: the projection is exact up to floating-point summation error,
    # ~N * eps * max(C_k) (order 1e-11 on the reference fixture).  1e-9 *
    # max(C_k) leaves several orders of headroom while being 1000x tighter
    # than the pre-fix 1e-6 * max(C_k).
    eq_resid <- abs(sum(alpha - alpha_star))
    eq_tol   <- 1e-9 * max(C_k)
    if (eq_resid > eq_tol) {
      stop(sprintf(
        paste0("warm-start equality residual %.2e exceeds tolerance %.2e ",
               "after projection. SMO conserves sum(alpha - alpha_star), so ",
               "an infeasible start would be carried through to the returned ",
               "solution. Pass warm_start_check = FALSE to override."),
        eq_resid, eq_tol))
    }
    if (any(alpha      < 0 - 1e-12) || any(alpha      > C_k + 1e-12))
      stop("warm-start alpha violates per-sample box after projection.")
    if (any(alpha_star < 0 - 1e-12) || any(alpha_star > C_k + 1e-12))
      stop("warm-start alpha_star violates per-sample box after projection.")
  }

  list(alpha = alpha, alpha_star = alpha_star)
}
