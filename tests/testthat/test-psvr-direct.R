## Bit-identical golden snapshots for direct psvr_mape() / psvr_rmspe() calls
## (F1 Step 4; repointed from psvr() in API-redesign stage 5).
##
## 12 tests covering 4 (family x symmetry) x 3 kernels.  Same fixture and HP as
## test-bit-identical.R; tolerance 1e-10. These snapshots should match
## the parsnip-pipeline snapshots in test-bit-identical.R byte-for-byte
## since both entry points route through the same fitters.
##
## The test_that() NAMES ARE FROZEN: expect_snapshot_value() keys on them, so
## rewording one regenerates its entry in _snaps/psvr-direct.md and destroys
## the proof that the numerics did not move. They still say "psvr mape" and
## "sym=+1" for that reason, not because the calls do.

# ---- Shared fixture (same as test-bit-identical.R) -----------------------
make_fixture <- function() {
  set.seed(2026)
  X      <- matrix(stats::rnorm(50 * 5), 50, 5,
                   dimnames = list(NULL, paste0("V", 1:5)))
  y      <- stats::rlnorm(50, meanlog = 0, sdlog = 0.5)
  X_test <- matrix(stats::rnorm(20 * 5), 20, 5,
                   dimnames = list(NULL, paste0("V", 1:5)))
  list(X = X, y = y, X_test = X_test)
}

HP <- list(
  C            = 10,
  eps          = 5,
  gamma        = 100,
  rbf_sigma    = 1,
  degree       = 2L,
  scale_factor = 1,
  a            = 1L
)

# ---- 12 tests: family x symmetry x kernel --------------------------------
# HP$a = 1L is retained but no longer read: the calls below now say
# sym_type = "even" literally, which is what the parsnip half of
# test-bit-identical.R has said since stage 1. The two halves finally agree
# on one vocabulary.
# Sym + linear has K_sym = 0 by construction, collapsing predictions to b
# (matches the analogous behavior in test-bit-identical.R).

test_that("psvr mape / no sym / RBF — direct golden", {
  skip_on_cran()
  fx <- make_fixture()
  K  <- make_kernel("rbf", sigma = HP$rbf_sigma)
  fit <- psvr_mape(fx$X, fx$y, kernel = K, C = HP$C, eps = HP$eps)
  preds <- predict(fit, fx$X_test)
  expect_snapshot_value(preds, style = "serialize", tolerance = 1e-10)
})

test_that("psvr mape / no sym / poly — direct golden", {
  skip_on_cran()
  fx <- make_fixture()
  K  <- make_kernel("polynomial", degree = HP$degree, coef0 = HP$scale_factor)
  fit <- psvr_mape(fx$X, fx$y, kernel = K, C = HP$C, eps = HP$eps)
  preds <- predict(fit, fx$X_test)
  expect_snapshot_value(preds, style = "serialize", tolerance = 1e-10)
})

test_that("psvr mape / no sym / linear — direct golden", {
  skip_on_cran()
  fx <- make_fixture()
  K  <- make_kernel("linear")
  fit <- psvr_mape(fx$X, fx$y, kernel = K, C = HP$C, eps = HP$eps)
  preds <- predict(fit, fx$X_test)
  expect_snapshot_value(preds, style = "serialize", tolerance = 1e-10)
})

test_that("psvr mape / sym=+1 / RBF — direct golden", {
  skip_on_cran()
  fx <- make_fixture()
  K  <- make_kernel("rbf", sigma = HP$rbf_sigma)
  fit <- psvr_mape(fx$X, fx$y, sym_type = "even",
                   kernel = K, C = HP$C, eps = HP$eps)
  preds <- predict(fit, fx$X_test)
  expect_snapshot_value(preds, style = "serialize", tolerance = 1e-10)
})

test_that("psvr mape / sym=+1 / poly — direct golden", {
  skip_on_cran()
  fx <- make_fixture()
  K  <- make_kernel("polynomial", degree = HP$degree, coef0 = HP$scale_factor)
  fit <- psvr_mape(fx$X, fx$y, sym_type = "even",
                   kernel = K, C = HP$C, eps = HP$eps)
  preds <- predict(fit, fx$X_test)
  expect_snapshot_value(preds, style = "serialize", tolerance = 1e-10)
})

test_that("psvr mape / sym=+1 / linear — direct golden", {
  skip_on_cran()
  fx <- make_fixture()
  K  <- make_kernel("linear")
  fit <- psvr_mape(fx$X, fx$y, sym_type = "even",
                   kernel = K, C = HP$C, eps = HP$eps)
  preds <- predict(fit, fx$X_test)
  expect_snapshot_value(preds, style = "serialize", tolerance = 1e-10)
})

test_that("psvr rmspe / no sym / RBF — direct golden", {
  skip_on_cran()
  fx <- make_fixture()
  K  <- make_kernel("rbf", sigma = HP$rbf_sigma)
  fit <- psvr_rmspe(fx$X, fx$y, kernel = K, gamma = HP$gamma)
  preds <- predict(fit, fx$X_test)
  expect_snapshot_value(preds, style = "serialize", tolerance = 1e-10)
})

test_that("psvr rmspe / no sym / poly — direct golden", {
  skip_on_cran()
  fx <- make_fixture()
  K  <- make_kernel("polynomial", degree = HP$degree, coef0 = HP$scale_factor)
  fit <- psvr_rmspe(fx$X, fx$y, kernel = K, gamma = HP$gamma)
  preds <- predict(fit, fx$X_test)
  expect_snapshot_value(preds, style = "serialize", tolerance = 1e-10)
})

test_that("psvr rmspe / no sym / linear — direct golden", {
  skip_on_cran()
  fx <- make_fixture()
  K  <- make_kernel("linear")
  fit <- psvr_rmspe(fx$X, fx$y, kernel = K, gamma = HP$gamma)
  preds <- predict(fit, fx$X_test)
  expect_snapshot_value(preds, style = "serialize", tolerance = 1e-10)
})

test_that("psvr rmspe / sym=+1 / RBF — direct golden", {
  skip_on_cran()
  fx <- make_fixture()
  K  <- make_kernel("rbf", sigma = HP$rbf_sigma)
  fit <- psvr_rmspe(fx$X, fx$y, sym_type = "even",
                    kernel = K, gamma = HP$gamma)
  preds <- predict(fit, fx$X_test)
  expect_snapshot_value(preds, style = "serialize", tolerance = 1e-10)
})

test_that("psvr rmspe / sym=+1 / poly — direct golden", {
  skip_on_cran()
  fx <- make_fixture()
  K  <- make_kernel("polynomial", degree = HP$degree, coef0 = HP$scale_factor)
  fit <- psvr_rmspe(fx$X, fx$y, sym_type = "even",
                    kernel = K, gamma = HP$gamma)
  preds <- predict(fit, fx$X_test)
  expect_snapshot_value(preds, style = "serialize", tolerance = 1e-10)
})

test_that("psvr rmspe / sym=+1 / linear — direct golden", {
  skip_on_cran()
  fx <- make_fixture()
  K  <- make_kernel("linear")
  fit <- psvr_rmspe(fx$X, fx$y, sym_type = "even",
                    kernel = K, gamma = HP$gamma)
  preds <- predict(fit, fx$X_test)
  expect_snapshot_value(preds, style = "serialize", tolerance = 1e-10)
})

# ---- F5 field-presence contract --------------------------------------------
# Two of the three blocks that used to live here are gone, for two different
# reasons. Both were written against psvr_fit and neither survives the split
# to family-specific classes:
#
#   * "RMSPE: alpha_star and beta are NULL" asserted SHAPE UNIFORMITY -- that
#     one class serving both families materialises the inapplicable slots as
#     NULL. With four family-specific classes there is no shared shape for the
#     proposition to be about, so it does not become false, it loses its
#     referent. (Contrast test-spectral.R's expect_null block, which asserts
#     BEHAVIOUR -- that Models 1/3/4 run no spectral guard. That is still true;
#     only its encoding moved from a NULL slot to an absent field, so it was
#     repointed rather than deleted.)
#   * "solver_meta$iters and converged propagate from SMO" is duplicated
#     verbatim by test-mape-svr.R:47-48 once the vocabulary is flattened.
#
# What survives is the one assertion neither file duplicates: that beta is the
# pruned alpha - alpha_star on the support-vector indices.

test_that("psvr_mape: alpha/alpha_star length N, beta is the pruned difference", {
  fx <- make_fixture()
  K  <- make_kernel("rbf", sigma = HP$rbf_sigma)
  fit <- suppressWarnings(
    psvr_mape(fx$X, fx$y, kernel = K, C = HP$C, eps = HP$eps)
  )
  N <- nrow(fx$X)
  expect_length(fit$alpha,      N)
  expect_length(fit$alpha_star, N)
  # On SV indices, beta equals alpha - alpha_star within numerical tol.
  beta_full <- fit$alpha - fit$alpha_star
  sv_idx <- which(abs(beta_full) > 1e-5)
  expect_equal(beta_full[sv_idx], fit$beta, tolerance = 1e-10)
})
