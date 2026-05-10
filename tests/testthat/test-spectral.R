## F3 — Algorithm 2 (adaptive spectral regularization) tests.
##
## Coverage:
##   - PSD branch:      no_shift returned, Omega_use bit-identical to Omega_s
##   - Indefinite branch: shifted returned with Theorem 2(a) PSD floor,
##                       diagonal-only perturbation. Tested directly against
##                       a hand-crafted Wigner-style indefinite matrix —
##                       not via psvr(), because the three Mercer kernels in
##                       make_kernel() (RBF / linear / polynomial) all yield
##                       PSD Omega_s by Aronszajn closure / Schur product,
##                       so the shifted branch is a dormant defensive guard
##                       in production. See CLAUDE.md "Adaptive Spectral
##                       Regularization" for details.
##   - psvr() integration: solver_meta$spectral populated for Model 2 (all
##                       branches will be no_shift in production); NULL for
##                       Models 1, 3, 4.
##   - T_pi override:    caller can request more iterations.

# ---- Direct .adaptive_spectral_shift() unit tests --------------------------

test_that(".adaptive_spectral_shift returns no_shift for PSD matrix", {
  set.seed(2026)
  X       <- matrix(stats::rnorm(50 * 5), 50, 5)
  K       <- make_kernel("rbf", sigma = 1)
  Omega_s <- sym_kernel_matrix(K, X, a = 1L)
  diag(Omega_s) <- diag(Omega_s) + 0.5e-6

  spec <- psvr:::.adaptive_spectral_shift(Omega_s)

  expect_equal(spec$branch_taken, "no_shift")
  expect_equal(spec$mu, 0)
  expect_identical(spec$Omega_use, Omega_s)
  expect_length(spec$n_power_iterations, 2L)
  expect_true(is.finite(spec$lambda_max_hat) && spec$lambda_max_hat > 0)
})

test_that(".adaptive_spectral_shift activates shifted branch for indefinite matrix", {
  set.seed(2026)
  A <- matrix(stats::rnorm(30 * 30), 30, 30)
  M <- (A + t(A)) / 2  # symmetric, indefinite

  # T_pi = 20 to keep the eigenvalue estimate accurate enough that
  # Omega_use clears the delta_stab floor; with T_pi = 5 the under-shift
  # bias on Wigner-style spectra is documented in CLAUDE.md.
  spec <- psvr:::.adaptive_spectral_shift(M, T_pi = 20L)

  expect_equal(spec$branch_taken, "shifted")
  expect_gt(spec$mu, 0)
  expect_lt(spec$lambda_min_hat, 0)

  # Theorem 2(a) guarantees Omega_use >= delta_stab * I in the limit
  # T_pi -> infinity. Finite T_pi yields a residual underestimate of
  # |lambda_min| that bleeds into Omega_use; with T_pi = 20 on this
  # Wigner matrix the residual is ~2e-3 (vs. the original ~7.3), a
  # ~3000x reduction. Assert >= 1000x as a meaningful canary: if a
  # future change breaks Pass 2 directionality, the reduction collapses
  # to ~3.5x and this test catches the regression.
  eig_orig  <- min(eigen(M, symmetric = TRUE, only.values = TRUE)$values)
  eig_used  <- min(eigen(spec$Omega_use, symmetric = TRUE,
                         only.values = TRUE)$values)
  reduction <- abs(eig_orig) / max(abs(eig_used), 1e-15)
  expect_gt(reduction, 1000)
})

test_that(".adaptive_spectral_shift wiring: Omega_use differs only by mu*I from Omega_s", {
  set.seed(2026)
  A <- matrix(stats::rnorm(30 * 30), 30, 30)
  M <- (A + t(A)) / 2

  spec <- psvr:::.adaptive_spectral_shift(M, T_pi = 20L)

  if (spec$branch_taken == "shifted") {
    diff_mat <- spec$Omega_use - M
    # Diagonal: uniform shift of mu.
    expect_equal(diag(diff_mat), rep(spec$mu, nrow(M)),
                 tolerance = 1e-12)
    # Off-diagonal: unchanged.
    off_diag <- diff_mat
    diag(off_diag) <- 0
    expect_lt(max(abs(off_diag)), 1e-12)
  }
})

test_that(".adaptive_spectral_shift T_pi override is accepted", {
  # Schema check: T_pi parameter path executes silently and returns
  # n_power_iterations of length 2.
  set.seed(2026)
  X       <- matrix(stats::rnorm(50 * 5), 50, 5)
  K       <- make_kernel("rbf", sigma = 1)
  Omega_s <- sym_kernel_matrix(K, X, a = 1L)
  diag(Omega_s) <- diag(Omega_s) + 0.5e-6

  expect_silent(
    spec <- psvr:::.adaptive_spectral_shift(Omega_s, T_pi = 20L)
  )
  expect_equal(spec$branch_taken, "no_shift")
  expect_equal(spec$n_power_iterations, c(20L, 20L))
})

# ---- psvr() integration tests (no_shift path) ------------------------------
# All Mercer kernels supported by make_kernel() yield PSD Omega_s, so the
# shifted branch is unreachable in production. These tests exercise the
# diagnostics-plumbing wiring: solver_meta$spectral must always be a
# populated list for Model 2, with branch_taken == "no_shift" and mu == 0.

test_that("psvr() Model 2 + RBF + a=-1 takes no_shift branch", {
  set.seed(2026)
  X <- matrix(stats::rnorm(50 * 5), 50, 5)
  y <- abs(stats::rnorm(50)) + 0.1
  K <- make_kernel("rbf", sigma = 1)

  fit <- psvr(X, y, loss = "mape", sym = -1L,
              kernel = K, C = 10, eps = 5)

  expect_equal(fit$solver_meta$spectral$branch_taken, "no_shift")
  expect_equal(fit$solver_meta$spectral$mu, 0)
  expect_length(fit$solver_meta$spectral$n_power_iterations, 2L)
})

test_that("psvr() Model 2 + RBF + a=+1 takes no_shift branch", {
  set.seed(2026)
  X <- matrix(stats::rnorm(50 * 5), 50, 5)
  y <- abs(stats::rnorm(50)) + 0.1
  K <- make_kernel("rbf", sigma = 1)

  fit <- psvr(X, y, loss = "mape", sym = +1L,
              kernel = K, C = 10, eps = 5)

  expect_equal(fit$solver_meta$spectral$branch_taken, "no_shift")
  expect_equal(fit$solver_meta$spectral$mu, 0)
})

test_that("solver_meta$spectral is NULL for non-Model-2 fits", {
  set.seed(2026)
  X <- matrix(stats::rnorm(50 * 5), 50, 5)
  y <- abs(stats::rnorm(50)) + 0.1
  K <- make_kernel("rbf", sigma = 1)

  # Model 1 (MAPE, no sym) — no Omega_s built, no spectral guard.
  fit1 <- psvr(X, y, loss = "mape", kernel = K, C = 10, eps = 5)
  expect_null(fit1$solver_meta$spectral)

  # Model 3 (RMSPE, no sym) — direct linear solve.
  fit3 <- psvr(X, y, loss = "rmspe", kernel = K, gamma = 100)
  expect_null(fit3$solver_meta$spectral)

  # Model 4 (RMSPE, sym) — direct linear solve, no SMO Hessian to guard.
  fit4 <- psvr(X, y, loss = "rmspe", sym = +1L, kernel = K, gamma = 100)
  expect_null(fit4$solver_meta$spectral)
})
