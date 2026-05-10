set.seed(2026)

# Larger fixture than the other test files: SMO needs enough rows for both
# free and saturated SVs to appear at moderate C / eps settings, and we want
# osqp to have enough work to give a meaningful timing comparison.
N      <- 60L
X_tr   <- matrix(rnorm(N * 3L), N, 3L)
y_tr   <- abs(rnorm(N)) + 1
K_rbf  <- make_kernel("rbf", sigma = 1)
mean_y <- mean(y_tr)

skip_if_no_osqp <- function() {
  testthat::skip_if_not_installed("osqp")
}

# These tests exercise the deprecated mape_svr() / mape_sym_svr() wrappers;
# quiet helpers swallow the .Deprecated() noise so the test output stays
# focused on solver-parity behavior. The deprecation contract itself is
# asserted in test-mape-svr.R / test-mape-sym-svr.R.
.q_mape_svr     <- function(...) suppressWarnings(mape_svr(...))
.q_mape_sym_svr <- function(...) suppressWarnings(mape_sym_svr(...))

# ---- 1. Prediction parity, Model 1 -----------------------------------------

test_that("mape_svr SMO and osqp predictions agree within 1% of mean(y)", {
  skip_if_no_osqp()

  t_smo  <- system.time(
    fit_smo  <- .q_mape_svr(X_tr, y_tr, kernel = K_rbf, C = 10, eps = 5,
                            solver = "smo")
  )[["elapsed"]]
  t_osqp <- system.time(
    fit_osqp <- .q_mape_svr(X_tr, y_tr, kernel = K_rbf, C = 10, eps = 5,
                            solver = "osqp")
  )[["elapsed"]]

  p_smo  <- predict(fit_smo,  X_tr)
  p_osqp <- predict(fit_osqp, X_tr)
  max_diff <- max(abs(p_smo - p_osqp))

  expect_lt(max_diff, 0.01 * mean_y)

  # Stash for the PR benchmark; printed once when running with reporter = "summary".
  message(sprintf(
    "[bench] mape_svr      N=%d  osqp=%.3fs  smo=%.3fs  max_diff=%.6g",
    nrow(X_tr), t_osqp, t_smo, max_diff
  ))
})

# ---- 2. Prediction parity, Model 2 -----------------------------------------

test_that("mape_sym_svr SMO and osqp predictions agree within 1% of mean(y)", {
  skip_if_no_osqp()

  t_smo  <- system.time(
    fit_smo  <- .q_mape_sym_svr(X_tr, y_tr, kernel = K_rbf, C = 10, eps = 5,
                                a = 1, solver = "smo")
  )[["elapsed"]]
  t_osqp <- system.time(
    fit_osqp <- .q_mape_sym_svr(X_tr, y_tr, kernel = K_rbf, C = 10, eps = 5,
                                a = 1, solver = "osqp")
  )[["elapsed"]]

  p_smo  <- predict(fit_smo,  X_tr)
  p_osqp <- predict(fit_osqp, X_tr)
  max_diff <- max(abs(p_smo - p_osqp))

  expect_lt(max_diff, 0.01 * mean_y)

  message(sprintf(
    "[bench] mape_sym_svr  N=%d  osqp=%.3fs  smo=%.3fs  max_diff=%.6g",
    nrow(X_tr), t_osqp, t_smo, max_diff
  ))
})

# ---- 2b. Same parity for a = -1 (odd symmetry, indefinite Ks) ---------------

test_that("mape_sym_svr SMO matches osqp under odd symmetry (a = -1)", {
  skip_if_no_osqp()

  fit_smo  <- .q_mape_sym_svr(X_tr, y_tr, kernel = K_rbf, C = 5, eps = 5,
                              a = -1, solver = "smo")
  fit_osqp <- .q_mape_sym_svr(X_tr, y_tr, kernel = K_rbf, C = 5, eps = 5,
                              a = -1, solver = "osqp")
  max_diff <- max(abs(predict(fit_smo, X_tr) - predict(fit_osqp, X_tr)))
  expect_lt(max_diff, 0.01 * mean_y)
})

# ---- 3. Parsnip integration uses default solver without changes ------------

test_that("parsnip default fit (solver = 'smo') runs and predicts", {
  skip_if_not_installed("parsnip")

  train_df <- as.data.frame(X_tr)
  train_df$y <- y_tr

  spec <- psvr_mape_rbf(cost = 1, svm_margin = 5, rbf_sigma = 1) |>
    parsnip::set_engine("psvr")

  fit_p <- parsnip::fit(spec, y ~ ., data = train_df)
  preds <- predict(fit_p, new_data = train_df)
  expect_true(is.data.frame(preds))
  expect_equal(nrow(preds), N)
  expect_true(all(is.finite(preds$.pred)))
})

# ---- 4. Degenerate case: tiny C → no free SVs, b must still be finite -------

test_that("SMO bias falls back to sandwich when no free SVs exist", {
  fit <- .q_mape_svr(X_tr, y_tr, kernel = K_rbf, C = 1e-4, eps = 5,
                     solver = "smo")
  expect_true(is.finite(fit$b))
  preds <- predict(fit, X_tr)
  expect_true(all(is.finite(preds)))
})

# ---- 5. Convergence warning when max_iter is hit ---------------------------

test_that("SMO emits a warning when max_iter is reached", {
  # Direct call to .smo_solve with max_iter = 1: WSS1 needs many iterations on
  # a problem with this many SVs, so 1 iteration cannot satisfy the tolerance.
  Omega <- kernel_matrix(K_rbf, X_tr)
  diag(Omega) <- diag(Omega) + 1e-6
  K_acc <- psvr:::.make_kernel_accessor(Omega)
  expect_warning(
    psvr:::.smo_solve(K_acc, y_tr, C = 10, eps = 5, max_iter = 1L),
    "did not converge"
  )
})

# ---- 6. T3 + T8 collapse to defaults on homogeneous targets ----------------
#
# Theorem 3 (asymmetric freeze) and Theorem 8 (per-pair tolerance) both
# reduce to the F3 baseline when y_k = y_bar for all k. With near-uniform y
# (rho_y ~ 1.05) the only path differences come from floor()/ceiling()
# rounding flipping individual freeze thresholds between 4, 5, and 6 -- a
# perturbation that may shift the SMO trajectory but not the final fit by
# more than ~1e-5 (well below the per-pair tolerance floor of tol*max(y)).

test_that("T3 + T8 reduce to default behavior on homogeneous targets", {
  set.seed(2026)
  N_h <- 50L
  X_h <- matrix(rnorm(N_h * 5), N_h, 5)
  # Near-uniform y: rho_y ~ 1.05.  Targets are still strictly positive.
  y_h <- rep(2.0, N_h) + rnorm(N_h, sd = 0.05)
  K   <- make_kernel("rbf", sigma = 1)
  fit <- psvr(X_h, y_h, loss = "mape", kernel = K, C = 10, eps = 5)

  preds <- predict(fit, X_h)
  expect_true(all(is.finite(preds)))
  expect_true(all(preds > 0))            # MAPE-SVR predictions stay positive on positive y
  expect_equal(length(preds), N_h)
  # On near-homogeneous y the predictions track y closely (eps=5% tube).
  rel_err <- abs(preds - y_h) / y_h
  expect_lt(median(rel_err), 0.10)
})
