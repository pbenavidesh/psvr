# Tests for Theorem 5 (warm-start API) and the .warm_start_init() helper.

set.seed(2026)
N      <- 40L
X_tr   <- matrix(rnorm(N * 3L), N, 3L)
y_tr   <- abs(rnorm(N)) + 1
K_rbf  <- make_kernel("rbf", sigma = 1)

# ---- 1. Algorithm 1 projection: feasibility on a feasible input ------------

test_that(".warm_start_init() projects to the feasible region", {
  C_k <- rep(10, 5L)
  # Already-feasible input: sum(alpha - alpha_star) = 0, all in [0, C_k].
  ws  <- psvr:::.warm_start_init(
    alpha_init      = c(1, 2, 0, 0, 1),
    alpha_star_init = c(0, 0, 2, 1, 1),
    N = 5L, C_k = C_k
  )
  expect_equal(sum(ws$alpha - ws$alpha_star), 0, tolerance = 1e-12)
  expect_true(all(ws$alpha      >= 0 & ws$alpha      <= C_k))
  expect_true(all(ws$alpha_star >= 0 & ws$alpha_star <= C_k))
})

# A zero-violation input is already feasible, so the projection must be the
# identity on it (lambda = 0) -- nothing moves, in either direction.
test_that(".warm_start_init() leaves an already-feasible input untouched", {
  C_k <- rep(10, 6L)
  # Retained samples (1, 2, 3): converged from previous fold (sum balanced).
  # New samples (4, 5, 6): zero-filled (no info).
  alpha      <- c(3, 1, 0, 0, 0, 0)
  alpha_star <- c(0, 0, 4, 0, 0, 0)  # violation = 3 + 1 - 4 = 0
  ws <- psvr:::.warm_start_init(
    alpha_init      = alpha,
    alpha_star_init = alpha_star,
    N = 6L, C_k = C_k
  )
  # Previously-converged samples preserved exactly.
  expect_equal(ws$alpha[1:3],      alpha[1:3])
  expect_equal(ws$alpha_star[1:3], alpha_star[1:3])
  # Zero-filled samples remain at zero (no violation to absorb).
  expect_equal(ws$alpha[4:6],      numeric(3L))
  expect_equal(ws$alpha_star[4:6], numeric(3L))
})

test_that(".warm_start_init() absorbs a violation by minimum-norm projection", {
  # Rewritten at 0.0.2.9010. This test previously asserted the
  # new-samples-only shift (retained values untouched, the whole violation
  # pushed onto samples 3:5). That scheme is gone -- it destroyed 100% of the
  # correction whenever the violation was positive, because new samples sit at
  # alpha = 0 and the box clip undid the shift. The projection now ranges over
  # all 2N variables, so retained values move too; the assertions below pin
  # the exact minimum-norm answer rather than the old shift.
  C_k <- rep(10, 5L)
  # Retained samples (1, 2): alpha = 3, alpha_star = 5. Violation = 3 - 5 = -2.
  alpha      <- c(3, 0, 0, 0, 0)
  alpha_star <- c(0, 5, 0, 0, 0)
  ws <- psvr:::.warm_start_init(
    alpha_init      = alpha,
    alpha_star_init = alpha_star,
    N = 5L, C_k = C_k
  )
  # With lambda < 0 nothing hits a bound, so g(lambda) = -2 - 6*lambda and
  # lambda = -1/3: alpha_k = alpha0_k + 1/3, alpha*_k = alpha*0_k - 1/3.
  expect_equal(ws$alpha,      c(3 + 1/3, rep(1/3, 4L)), tolerance = 1e-12)
  expect_equal(ws$alpha_star, c(0, 5 - 1/3, 0, 0, 0),   tolerance = 1e-12)
  # Equality constraint satisfied at rounding level.
  expect_lt(abs(sum(ws$alpha - ws$alpha_star)), 1e-12)
  # Minimum-norm: no feasible point is closer to the input than this one.
  proj_dist <- sum((ws$alpha - alpha)^2) + sum((ws$alpha_star - alpha_star)^2)
  expect_equal(proj_dist, 6 * (1/3)^2, tolerance = 1e-12)
})

test_that(".warm_start_init() kills a POSITIVE violation (the pre-fix defect)", {
  # Regression guard for the bug fixed at 0.0.2.9010: a positive violation
  # used to be shifted onto new samples whose alpha was 0, driving them
  # negative so the box clip restored every one to 0 and the residual
  # survived intact. SMO conserves sum(alpha - alpha*), so that infeasible
  # start propagated into the returned solution.
  C_k <- rep(10, 5L)
  alpha      <- c(6, 4, 0, 0, 0)   # violation = +10, strictly positive
  alpha_star <- c(0, 0, 0, 0, 0)
  ws <- psvr:::.warm_start_init(
    alpha_init      = alpha,
    alpha_star_init = alpha_star,
    N = 5L, C_k = C_k
  )
  expect_lt(abs(sum(ws$alpha - ws$alpha_star)), 1e-12)
  expect_true(all(ws$alpha >= 0 & ws$alpha <= C_k))
  expect_true(all(ws$alpha_star >= 0 & ws$alpha_star <= C_k))
})

test_that(".warm_start_init() is exact when new samples alone cannot absorb it", {
  # Independent reason the new-samples-only restriction had to go: with
  # retained values held fixed the new block must absorb the entire
  # violation, which is infeasible when |violation| exceeds its capacity --
  # and degenerately so when there are no new samples at all.
  C_k <- rep(10, 4L)
  alpha      <- c(10, 10, 0, 0)   # violation = +20
  alpha_star <- c(0, 0, 0, 0)
  # n_new = 1, capacity of the new block = C_k = 10 < 20.
  ws <- psvr:::.warm_start_init(
    alpha_init = alpha, alpha_star_init = alpha_star,
    N = 4L, C_k = C_k
  )
  expect_lt(abs(sum(ws$alpha - ws$alpha_star)), 1e-12)

  # n_new = 0: the old code fell back to a uniform shift over N; the exact
  # projection needs no special case.
  ws0 <- psvr:::.warm_start_init(
    alpha_init = alpha, alpha_star_init = alpha_star,
    N = 4L, C_k = C_k
  )
  expect_lt(abs(sum(ws0$alpha - ws0$alpha_star)), 1e-12)
})

test_that(".warm_start_init() clips infeasible inputs into the box", {
  C_k <- rep(2, 4L)
  ws  <- psvr:::.warm_start_init(
    alpha_init      = c(5, -1, 0, 0),
    alpha_star_init = c(0, 0, 5, -1),
    N = 4L, C_k = C_k,
    warm_start_check = FALSE  # equality residual may exceed tol after clipping
  )
  expect_true(all(ws$alpha      >= 0 & ws$alpha      <= C_k))
  expect_true(all(ws$alpha_star >= 0 & ws$alpha_star <= C_k))
})

test_that(".warm_start_init() zero input is feasible", {
  ws <- psvr:::.warm_start_init(NULL, NULL, 10L, rep(5, 10L))
  expect_equal(ws$alpha,      numeric(10L))
  expect_equal(ws$alpha_star, numeric(10L))
})

# ---- 2. Warm-start fit reaches the same optimum --------------------------

test_that("psvr_mape(alpha_init = converged α) matches cold-start within tol", {
  fit_cold <- psvr_mape(X_tr, y_tr, kernel = K_rbf,
                        C = 10, eps = 5)
  # A converged input already satisfies the equality constraint, so the
  # projection is the identity and the solver restarts at its own optimum.
  fit_warm <- psvr_mape(X_tr, y_tr, kernel = K_rbf,
                        C = 10, eps = 5,
                        alpha_init      = fit_cold$alpha,
                        alpha_star_init = fit_cold$alpha_star)

  p_cold <- predict(fit_cold, X_tr)
  p_warm <- predict(fit_warm, X_tr)
  expect_lt(max(abs(p_cold - p_warm)), 1e-3 * mean(y_tr))
  # Warm-start from converged state should not need more iterations than
  # the original cold-start run.
  expect_lte(fit_warm$iterations, fit_cold$iterations)
})

# ---- 3. Strict error: rmspe + warm-start --------------------------------

# psvr_rmspe() has no alpha_init / alpha_star_init formals at all, so a bare
# check_dots_empty() would report only that the name is unknown. The targeted
# guard keeps the reason, and this block keeps pinning it. The wording is a
# NEVER, not a NOT YET: LS-SVR is one linear-system solve, so there is no SMO
# state to carry over and never will be.
test_that("psvr_rmspe() rejects warm-start vectors with the reason", {
  expect_error(
    psvr_rmspe(X_tr, y_tr, kernel = K_rbf, gamma = 100,
               alpha_init = numeric(N)),
    "Warm-start is not supported"
  )
  expect_error(
    psvr_rmspe(X_tr, y_tr, kernel = K_rbf, gamma = 100,
               alpha_star_init = numeric(N)),
    "Warm-start is not supported"
  )
})

# ---- 4. Length-mismatch on warm-start vectors ----------------------------

test_that("psvr_mape() rejects warm-start vectors of wrong length", {
  expect_error(
    psvr_mape(X_tr, y_tr, kernel = K_rbf, C = 10, eps = 5,
              alpha_init = numeric(N + 1L)),
    "length nrow\\(X\\)"
  )
  expect_error(
    psvr_mape(X_tr, y_tr, kernel = K_rbf, C = 10, eps = 5,
              alpha_star_init = numeric(N - 1L)),
    "length nrow\\(X\\)"
  )
  expect_error(
    psvr_mape(X_tr, y_tr, kernel = K_rbf, C = 10, eps = 5,
              alpha_init = c(NA_real_, numeric(N - 1L))),
    "finite numeric"
  )
})

# ---- 5. warm_start_check = FALSE skips post-projection assertions --------

test_that("warm_start_check = FALSE bypasses the feasibility check", {
  # Wildly out-of-box input. Since 0.0.2.9010 the exact projection handles
  # this case too (the clip in x(lambda) enforces the box regardless of where
  # x0 sits), so the fit would run either way; the test still pins that
  # warm_start_check = FALSE is a working escape hatch now that the check
  # raises an error rather than a warning.
  bad_init <- rep(1e3, N)
  expect_no_error(
    psvr_mape(X_tr, y_tr, kernel = K_rbf, C = 10, eps = 5,
              alpha_init = bad_init, alpha_star_init = numeric(N),
              warm_start_check = FALSE)
  )
})
