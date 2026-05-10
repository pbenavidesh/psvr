## Bit-identical golden snapshots — pre-refactor baseline (F1 Step 1).
##
## These tests serialize predict() outputs from the current implementation
## under fixed seeds and hyperparameters. The post-F1 implementation must
## reproduce these byte-for-byte (tolerance 1e-10) — any larger drift
## signals an unintended numerical change.
##
## Coverage (16 tests):
##   - 4 direct-fitter tests, one per loss x sym, RBF kernel
##   - 12 parsnip-pipeline tests, 4 (loss x sym) x 3 kernels

library(parsnip)

# ---- Fixture (called inside each test for full determinism) --------------
make_fixture <- function() {
  set.seed(2026)
  X      <- matrix(stats::rnorm(50 * 5), 50, 5,
                   dimnames = list(NULL, paste0("V", 1:5)))
  y      <- stats::rlnorm(50, meanlog = 0, sdlog = 0.5)
  X_test <- matrix(stats::rnorm(20 * 5), 20, 5,
                   dimnames = list(NULL, paste0("V", 1:5)))
  list(X       = X,
       y       = y,
       X_test  = X_test,
       df_test = as.data.frame(X_test))
}

# Hyperparameters fixed across all tests.
HP <- list(
  C            = 10,
  eps          = 5,
  gamma        = 100,
  rbf_sigma    = 1,
  degree       = 2L,
  scale_factor = 1,
  a            = 1L
)


# ---- 4 direct-fitter golden tests (RBF kernel) ---------------------------
# These tests intentionally call the deprecated wrappers to lock in the
# pre-refactor numerics. The .Deprecated() notice is asserted in the
# dedicated contract tests of test-mape-svr.R / test-rmspe-lssvr.R / etc.;
# here it is suppressed to keep snapshot-test output focused.

test_that("Model 1 mape_svr (RBF) — direct golden", {
  fx  <- make_fixture()
  K   <- make_kernel("rbf", sigma = HP$rbf_sigma)
  fit <- suppressWarnings(
    mape_svr(fx$X, fx$y, kernel = K, C = HP$C, eps = HP$eps)
  )
  preds <- predict(fit, fx$X_test)
  expect_snapshot_value(preds, style = "serialize", tolerance = 1e-10)
})

test_that("Model 2 mape_sym_svr (RBF) — direct golden", {
  fx  <- make_fixture()
  K   <- make_kernel("rbf", sigma = HP$rbf_sigma)
  fit <- suppressWarnings(
    mape_sym_svr(fx$X, fx$y, kernel = K,
                 C = HP$C, eps = HP$eps, a = HP$a)
  )
  preds <- predict(fit, fx$X_test)
  expect_snapshot_value(preds, style = "serialize", tolerance = 1e-10)
})

test_that("Model 3 rmspe_lssvr (RBF) — direct golden", {
  fx  <- make_fixture()
  K   <- make_kernel("rbf", sigma = HP$rbf_sigma)
  fit <- suppressWarnings(
    rmspe_lssvr(fx$X, fx$y, kernel = K, gamma = HP$gamma)
  )
  preds <- predict(fit, fx$X_test)
  expect_snapshot_value(preds, style = "serialize", tolerance = 1e-10)
})

test_that("Model 4 rmspe_sym_lssvr (RBF) — direct golden", {
  fx  <- make_fixture()
  K   <- make_kernel("rbf", sigma = HP$rbf_sigma)
  fit <- suppressWarnings(
    rmspe_sym_lssvr(fx$X, fx$y, kernel = K,
                    gamma = HP$gamma, a = HP$a)
  )
  preds <- predict(fit, fx$X_test)
  expect_snapshot_value(preds, style = "serialize", tolerance = 1e-10)
})


# ---- 12 parsnip-pipeline golden tests ------------------------------------
# Helper: extract numeric .pred vector to make snapshots invariant to any
# tibble-class metadata changes in future parsnip versions.
fit_and_predict <- function(spec, fx) {
  fit_obj <- parsnip::fit_xy(spec, x = fx$X, y = fx$y)
  predict(fit_obj, new_data = fx$df_test)$.pred
}

# Model 1: MAPE (no sym) ---------------------------------------------------

test_that("psvr_mape_rbf — parsnip golden", {
  fx   <- make_fixture()
  spec <- psvr_mape_rbf(cost = HP$C, svm_margin = HP$eps,
                        rbf_sigma = HP$rbf_sigma) |>
            set_engine("psvr")
  preds <- fit_and_predict(spec, fx)
  expect_snapshot_value(preds, style = "serialize", tolerance = 1e-10)
})

test_that("psvr_mape_poly — parsnip golden", {
  fx   <- make_fixture()
  spec <- psvr_mape_poly(cost = HP$C, svm_margin = HP$eps,
                         degree = HP$degree,
                         scale_factor = HP$scale_factor) |>
            set_engine("psvr")
  preds <- fit_and_predict(spec, fx)
  expect_snapshot_value(preds, style = "serialize", tolerance = 1e-10)
})

test_that("psvr_mape_linear — parsnip golden", {
  fx   <- make_fixture()
  spec <- psvr_mape_linear(cost = HP$C, svm_margin = HP$eps) |>
            set_engine("psvr")
  preds <- fit_and_predict(spec, fx)
  expect_snapshot_value(preds, style = "serialize", tolerance = 1e-10)
})

# Model 2: MAPE sym --------------------------------------------------------
# `sym_type` is a tunable arg only on the RBF spec; poly/linear take `a`
# as an engine arg (matches existing test-parsnip.R smoke tests).

test_that("psvr_mape_sym_rbf — parsnip golden", {
  fx   <- make_fixture()
  spec <- psvr_mape_sym_rbf(cost = HP$C, svm_margin = HP$eps,
                            rbf_sigma = HP$rbf_sigma,
                            sym_type = "even") |>
            set_engine("psvr")
  preds <- fit_and_predict(spec, fx)
  expect_snapshot_value(preds, style = "serialize", tolerance = 1e-10)
})

test_that("psvr_mape_sym_poly — parsnip golden", {
  fx   <- make_fixture()
  spec <- psvr_mape_sym_poly(cost = HP$C, svm_margin = HP$eps,
                             degree = HP$degree,
                             scale_factor = HP$scale_factor) |>
            set_engine("psvr", a = HP$a)
  preds <- fit_and_predict(spec, fx)
  expect_snapshot_value(preds, style = "serialize", tolerance = 1e-10)
})

test_that("psvr_mape_sym_linear — parsnip golden", {
  fx   <- make_fixture()
  spec <- psvr_mape_sym_linear(cost = HP$C, svm_margin = HP$eps) |>
            set_engine("psvr", a = HP$a)
  preds <- fit_and_predict(spec, fx)
  expect_snapshot_value(preds, style = "serialize", tolerance = 1e-10)
})

# Model 3: RMSPE (no sym) --------------------------------------------------

test_that("psvr_rmspe_rbf — parsnip golden", {
  fx   <- make_fixture()
  spec <- psvr_rmspe_rbf(cost = HP$gamma, rbf_sigma = HP$rbf_sigma) |>
            set_engine("psvr")
  preds <- fit_and_predict(spec, fx)
  expect_snapshot_value(preds, style = "serialize", tolerance = 1e-10)
})

test_that("psvr_rmspe_poly — parsnip golden", {
  fx   <- make_fixture()
  spec <- psvr_rmspe_poly(cost = HP$gamma, degree = HP$degree,
                          scale_factor = HP$scale_factor) |>
            set_engine("psvr")
  preds <- fit_and_predict(spec, fx)
  expect_snapshot_value(preds, style = "serialize", tolerance = 1e-10)
})

test_that("psvr_rmspe_linear — parsnip golden", {
  fx   <- make_fixture()
  spec <- psvr_rmspe_linear(cost = HP$gamma) |>
            set_engine("psvr")
  preds <- fit_and_predict(spec, fx)
  expect_snapshot_value(preds, style = "serialize", tolerance = 1e-10)
})

# Model 4: RMSPE sym -------------------------------------------------------

test_that("psvr_rmspe_sym_rbf — parsnip golden", {
  fx   <- make_fixture()
  spec <- psvr_rmspe_sym_rbf(cost = HP$gamma,
                             rbf_sigma = HP$rbf_sigma,
                             sym_type = "even") |>
            set_engine("psvr")
  preds <- fit_and_predict(spec, fx)
  expect_snapshot_value(preds, style = "serialize", tolerance = 1e-10)
})

test_that("psvr_rmspe_sym_poly — parsnip golden", {
  fx   <- make_fixture()
  spec <- psvr_rmspe_sym_poly(cost = HP$gamma, degree = HP$degree,
                              scale_factor = HP$scale_factor) |>
            set_engine("psvr", a = HP$a)
  preds <- fit_and_predict(spec, fx)
  expect_snapshot_value(preds, style = "serialize", tolerance = 1e-10)
})

test_that("psvr_rmspe_sym_linear — parsnip golden", {
  fx   <- make_fixture()
  spec <- psvr_rmspe_sym_linear(cost = HP$gamma) |>
            set_engine("psvr", a = HP$a)
  preds <- fit_and_predict(spec, fx)
  expect_snapshot_value(preds, style = "serialize", tolerance = 1e-10)
})
