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

# Round 2: new-samples-only projection preserves retained-sample alpha values
# exactly (rather than perturbing them with a uniform shift).
test_that(".warm_start_init() new-samples-only projection preserves retained values", {
  C_k <- rep(10, 6L)
  # Retained samples (1, 2, 3): converged from previous fold (sum balanced).
  # New samples (4, 5, 6): zero-filled (no info).
  alpha      <- c(3, 1, 0, 0, 0, 0)
  alpha_star <- c(0, 0, 4, 0, 0, 0)  # violation = 3 + 1 - 4 = 0
  ws <- psvr:::.warm_start_init(
    alpha_init      = alpha,
    alpha_star_init = alpha_star,
    N = 6L, C_k = C_k,
    new_mask = c(FALSE, FALSE, FALSE, TRUE, TRUE, TRUE)
  )
  # Retained samples preserved exactly.
  expect_equal(ws$alpha[1:3],      alpha[1:3])
  expect_equal(ws$alpha_star[1:3], alpha_star[1:3])
  # New samples remain at zero (no violation to absorb).
  expect_equal(ws$alpha[4:6],      numeric(3L))
  expect_equal(ws$alpha_star[4:6], numeric(3L))
})

test_that(".warm_start_init() new-samples-only absorbs violation correctly", {
  C_k <- rep(10, 5L)
  # Retained samples (1, 2): alpha=3, alpha_star=5. Violation = 3 - 5 = -2.
  # New samples (3, 4, 5): zero-filled.  With per_new = -2/3, the shift
  # alpha[new] -= per_new = +0.667 lands inside [0, C_k] without clipping,
  # so the residual safety net does NOT activate.
  alpha      <- c(3, 0, 0, 0, 0)
  alpha_star <- c(0, 5, 0, 0, 0)
  ws <- psvr:::.warm_start_init(
    alpha_init      = alpha,
    alpha_star_init = alpha_star,
    N = 5L, C_k = C_k,
    new_mask = c(FALSE, FALSE, TRUE, TRUE, TRUE)
  )
  # Retained sample values preserved exactly.
  expect_equal(ws$alpha[1L],      3)
  expect_equal(ws$alpha_star[2L], 5)
  # New samples 3, 4, 5 each absorbed +2/3 of the violation gap.
  expect_equal(ws$alpha[3:5], rep(2/3, 3L), tolerance = 1e-12)
  # Equality constraint exactly satisfied.
  expect_lt(abs(sum(ws$alpha - ws$alpha_star)), 1e-12)
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

test_that("psvr(alpha_init = converged α) matches cold-start within tol", {
  fit_cold <- psvr(X_tr, y_tr, loss = "mape", kernel = K_rbf,
                   C = 10, eps = 5)
  # Mark all samples as RETAINED (none new) so the new-samples-only Step 2
  # falls back to paper-uniform shift; the converged input should give
  # violation = 0 anyway.
  fit_warm <- psvr(X_tr, y_tr, loss = "mape", kernel = K_rbf,
                   C = 10, eps = 5,
                   alpha_init      = fit_cold$alpha,
                   alpha_star_init = fit_cold$alpha_star,
                   new_mask        = rep(FALSE, N))

  p_cold <- predict(fit_cold, X_tr)
  p_warm <- predict(fit_warm, X_tr)
  expect_lt(max(abs(p_cold - p_warm)), 1e-3 * mean(y_tr))
  # Warm-start from converged state should not need more iterations than
  # the original cold-start run.
  expect_lte(fit_warm$solver_meta$iters, fit_cold$solver_meta$iters)
})

# ---- 3. Strict error: rmspe + warm-start --------------------------------

test_that("psvr() rejects warm-start under loss = 'rmspe'", {
  expect_error(
    psvr(X_tr, y_tr, loss = "rmspe", kernel = K_rbf, gamma = 100,
         alpha_init = numeric(N)),
    "Warm-start is not supported"
  )
  expect_error(
    psvr(X_tr, y_tr, loss = "rmspe", kernel = K_rbf, gamma = 100,
         alpha_star_init = numeric(N)),
    "Warm-start is not supported"
  )
})

# ---- 4. Length-mismatch on warm-start vectors ----------------------------

test_that("psvr() rejects warm-start vectors of wrong length", {
  expect_error(
    psvr(X_tr, y_tr, loss = "mape", kernel = K_rbf, C = 10, eps = 5,
         alpha_init = numeric(N + 1L)),
    "length nrow\\(X\\)"
  )
  expect_error(
    psvr(X_tr, y_tr, loss = "mape", kernel = K_rbf, C = 10, eps = 5,
         alpha_star_init = numeric(N - 1L)),
    "length nrow\\(X\\)"
  )
  expect_error(
    psvr(X_tr, y_tr, loss = "mape", kernel = K_rbf, C = 10, eps = 5,
         alpha_init = c(NA_real_, numeric(N - 1L))),
    "finite numeric"
  )
})

# ---- 5. warm_start_check = FALSE skips post-projection assertions --------

test_that("warm_start_check = FALSE bypasses the feasibility check", {
  # Very-infeasible input that the projection cannot fix exactly; with
  # warm_start_check = FALSE we want the fit to still run (the SMO solver
  # will recover via its own iterations).
  bad_init <- rep(1e3, N)
  expect_no_error(
    psvr(X_tr, y_tr, loss = "mape", kernel = K_rbf, C = 10, eps = 5,
         alpha_init = bad_init, alpha_star_init = numeric(N),
         warm_start_check = FALSE)
  )
})
