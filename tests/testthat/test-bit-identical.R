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
# Every test in this file is a golden snapshot gate: the reference is a value
# recorded on one x86_64 toolchain, not a second computation, so there is no
# meaningful cross-platform counterpart to add alongside it. They run under
# skip_on_cran() and are pinned in CI by the one job that sets NOT_CRAN=true.
# See helper-fp-tiers.R for the policy and the measured deviations.
# DO NOT RENAME OR DELETE THESE FOUR test_that() BLOCKS.
# The names still say mape_svr / mape_sym_svr / rmspe_lssvr / rmspe_sym_lssvr,
# removed in stage 3 of the shape-B redesign. That is DELIBERATE:
# _snaps/bit-identical.md keys on test_that() names and is a protected F7.5
# baseline (MD5 46a4fa24). Rewording a name regenerates it and destroys the
# proof that the new API reproduces the pre-refactor numerics.
#
# These four are also byte-for-byte duplicates of test-psvr-direct.R's four RBF
# tests -- same call, same fixture, same 244-char serialized payload. Also
# DELIBERATE: they are the ONLY record tying the pre-refactor numerics to the
# OLD API. test-psvr-direct.R proves the current entry point matches a value; it
# does not prove mape_svr() ever produced it. Deleting these as "redundant"
# throws that away.
#
# Symmetry vocabulary, third and final spelling. The deleted wrappers took
# `a = HP$a` (HP$a = 1L); psvr() took `sym = HP$a` and mapped it via
# as.integer(); psvr_mape() / psvr_rmspe() take `sym_type = "even"` and map it
# via .sym_type_to_a() (R/parsnip.R:27). All three reach a = 1L, which is why
# the payloads below have never moved. The parsnip half of this file has said
# `sym_type = "even"` since stage 1, so the two halves now agree.

test_that("Model 1 mape_svr (RBF) — direct golden", {
  skip_on_cran()
  fx  <- make_fixture()
  K   <- make_kernel("rbf", sigma = HP$rbf_sigma)
  fit <- suppressWarnings(
    psvr_mape(fx$X, fx$y, kernel = K, C = HP$C, eps = HP$eps)
  )
  preds <- predict(fit, fx$X_test)
  expect_snapshot_value(preds, style = "serialize", tolerance = 1e-10)
})

test_that("Model 2 mape_sym_svr (RBF) — direct golden", {
  skip_on_cran()
  fx  <- make_fixture()
  K   <- make_kernel("rbf", sigma = HP$rbf_sigma)
  fit <- suppressWarnings(
    psvr_mape(fx$X, fx$y, sym_type = "even", kernel = K,
              C = HP$C, eps = HP$eps)
  )
  preds <- predict(fit, fx$X_test)
  expect_snapshot_value(preds, style = "serialize", tolerance = 1e-10)
})

test_that("Model 3 rmspe_lssvr (RBF) — direct golden", {
  skip_on_cran()
  fx  <- make_fixture()
  K   <- make_kernel("rbf", sigma = HP$rbf_sigma)
  fit <- suppressWarnings(
    psvr_rmspe(fx$X, fx$y, kernel = K, gamma = HP$gamma)
  )
  preds <- predict(fit, fx$X_test)
  expect_snapshot_value(preds, style = "serialize", tolerance = 1e-10)
})

test_that("Model 4 rmspe_sym_lssvr (RBF) — direct golden", {
  skip_on_cran()
  fx  <- make_fixture()
  K   <- make_kernel("rbf", sigma = HP$rbf_sigma)
  fit <- suppressWarnings(
    psvr_rmspe(fx$X, fx$y, sym_type = "even", kernel = K,
               gamma = HP$gamma)
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
  skip_on_cran()
  fx   <- make_fixture()
  spec <- psvr_mape_rbf(cost = HP$C, margin = HP$eps,
                        rbf_sigma = HP$rbf_sigma) |>
            set_engine("psvr")
  preds <- fit_and_predict(spec, fx)
  expect_snapshot_value(preds, style = "serialize", tolerance = 1e-10)
})

test_that("psvr_mape_poly — parsnip golden", {
  skip_on_cran()
  fx   <- make_fixture()
  spec <- psvr_mape_poly(cost = HP$C, margin = HP$eps,
                         degree = HP$degree,
                         scale_factor = HP$scale_factor) |>
            set_engine("psvr")
  preds <- fit_and_predict(spec, fx)
  expect_snapshot_value(preds, style = "serialize", tolerance = 1e-10)
})

test_that("psvr_mape_linear — parsnip golden", {
  skip_on_cran()
  fx   <- make_fixture()
  spec <- psvr_mape_linear(cost = HP$C, margin = HP$eps) |>
            set_engine("psvr")
  preds <- fit_and_predict(spec, fx)
  expect_snapshot_value(preds, style = "serialize", tolerance = 1e-10)
})

# Model 2: MAPE sym --------------------------------------------------------
# The test_that() NAMES below still say `psvr_mape_sym_*`. That is deliberate
# and load-bearing: _snaps/bit-identical.md keys on test_that() names and is a
# protected F7.5 baseline (MD5 46a4fa24). Rewording a name regenerates it and
# destroys the proof that the new API reproduces the pre-refactor numerics.
#
# The BODIES were repointed in the shape-B redesign (stage 2): the six
# `psvr_*_sym_*` specs were deleted and symmetry became the `sym_type` argument
# of the corresponding non-symmetric spec. `sym_type = "even"` maps to a = 1L
# via .sym_type_to_a(), reproducing what `a = HP$a` (HP$a = 1L) did before.
# Stage-1 equivalence tests confirmed all six pairs identical at tolerance = 0.

test_that("psvr_mape_sym_rbf — parsnip golden", {
  skip_on_cran()
  fx   <- make_fixture()
  spec <- psvr_mape_rbf(cost = HP$C, margin = HP$eps,
                        rbf_sigma = HP$rbf_sigma,
                        sym_type = "even") |>
            set_engine("psvr")
  preds <- fit_and_predict(spec, fx)
  expect_snapshot_value(preds, style = "serialize", tolerance = 1e-10)
})

test_that("psvr_mape_sym_poly — parsnip golden", {
  skip_on_cran()
  fx   <- make_fixture()
  spec <- psvr_mape_poly(cost = HP$C, margin = HP$eps,
                         degree = HP$degree,
                         scale_factor = HP$scale_factor,
                         sym_type = "even") |>
            set_engine("psvr")
  preds <- fit_and_predict(spec, fx)
  expect_snapshot_value(preds, style = "serialize", tolerance = 1e-10)
})

test_that("psvr_mape_sym_linear — parsnip golden", {
  skip_on_cran()
  fx   <- make_fixture()
  spec <- psvr_mape_linear(cost = HP$C, margin = HP$eps,
                           sym_type = "even") |>
            set_engine("psvr")
  preds <- fit_and_predict(spec, fx)
  expect_snapshot_value(preds, style = "serialize", tolerance = 1e-10)
})

# Model 3: RMSPE (no sym) --------------------------------------------------

test_that("psvr_rmspe_rbf — parsnip golden", {
  skip_on_cran()
  fx   <- make_fixture()
  spec <- psvr_rmspe_rbf(cost = HP$gamma, rbf_sigma = HP$rbf_sigma) |>
            set_engine("psvr")
  preds <- fit_and_predict(spec, fx)
  expect_snapshot_value(preds, style = "serialize", tolerance = 1e-10)
})

test_that("psvr_rmspe_poly — parsnip golden", {
  skip_on_cran()
  fx   <- make_fixture()
  spec <- psvr_rmspe_poly(cost = HP$gamma, degree = HP$degree,
                          scale_factor = HP$scale_factor) |>
            set_engine("psvr")
  preds <- fit_and_predict(spec, fx)
  expect_snapshot_value(preds, style = "serialize", tolerance = 1e-10)
})

test_that("psvr_rmspe_linear — parsnip golden", {
  skip_on_cran()
  fx   <- make_fixture()
  spec <- psvr_rmspe_linear(cost = HP$gamma) |>
            set_engine("psvr")
  preds <- fit_and_predict(spec, fx)
  expect_snapshot_value(preds, style = "serialize", tolerance = 1e-10)
})

# Model 4: RMSPE sym -------------------------------------------------------
# Names deliberately stale; bodies repointed. See the note above Model 2.

test_that("psvr_rmspe_sym_rbf — parsnip golden", {
  skip_on_cran()
  fx   <- make_fixture()
  spec <- psvr_rmspe_rbf(cost = HP$gamma,
                         rbf_sigma = HP$rbf_sigma,
                         sym_type = "even") |>
            set_engine("psvr")
  preds <- fit_and_predict(spec, fx)
  expect_snapshot_value(preds, style = "serialize", tolerance = 1e-10)
})

test_that("psvr_rmspe_sym_poly — parsnip golden", {
  skip_on_cran()
  fx   <- make_fixture()
  spec <- psvr_rmspe_poly(cost = HP$gamma, degree = HP$degree,
                          scale_factor = HP$scale_factor,
                          sym_type = "even") |>
            set_engine("psvr")
  preds <- fit_and_predict(spec, fx)
  expect_snapshot_value(preds, style = "serialize", tolerance = 1e-10)
})

test_that("psvr_rmspe_sym_linear — parsnip golden", {
  skip_on_cran()
  fx   <- make_fixture()
  spec <- psvr_rmspe_linear(cost = HP$gamma, sym_type = "even") |>
            set_engine("psvr")
  preds <- fit_and_predict(spec, fx)
  expect_snapshot_value(preds, style = "serialize", tolerance = 1e-10)
})
