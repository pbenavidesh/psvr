## Bit-identical golden snapshots for direct psvr() calls (F1 Step 4).
##
## 12 tests covering 4 (loss x sym) x 3 kernels.  Same fixture and HP as
## test-bit-identical.R; tolerance 1e-10. These snapshots should match
## the parsnip-pipeline snapshots in test-bit-identical.R byte-for-byte
## since psvr() routes through the same fitters.

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

# ---- 12 tests: psvr(loss × sym × kernel) ---------------------------------
# Sym + linear has K_sym = 0 by construction, collapsing predictions to b
# (matches the analogous behavior in test-bit-identical.R).

test_that("psvr mape / no sym / RBF — direct golden", {
  fx <- make_fixture()
  K  <- make_kernel("rbf", sigma = HP$rbf_sigma)
  fit <- psvr(fx$X, fx$y, loss = "mape", kernel = K, C = HP$C, eps = HP$eps)
  preds <- predict(fit, fx$X_test)
  expect_snapshot_value(preds, style = "serialize", tolerance = 1e-10)
})

test_that("psvr mape / no sym / poly — direct golden", {
  fx <- make_fixture()
  K  <- make_kernel("polynomial", degree = HP$degree, coef0 = HP$scale_factor)
  fit <- psvr(fx$X, fx$y, loss = "mape", kernel = K, C = HP$C, eps = HP$eps)
  preds <- predict(fit, fx$X_test)
  expect_snapshot_value(preds, style = "serialize", tolerance = 1e-10)
})

test_that("psvr mape / no sym / linear — direct golden", {
  fx <- make_fixture()
  K  <- make_kernel("linear")
  fit <- psvr(fx$X, fx$y, loss = "mape", kernel = K, C = HP$C, eps = HP$eps)
  preds <- predict(fit, fx$X_test)
  expect_snapshot_value(preds, style = "serialize", tolerance = 1e-10)
})

test_that("psvr mape / sym=+1 / RBF — direct golden", {
  fx <- make_fixture()
  K  <- make_kernel("rbf", sigma = HP$rbf_sigma)
  fit <- psvr(fx$X, fx$y, loss = "mape", sym = HP$a,
              kernel = K, C = HP$C, eps = HP$eps)
  preds <- predict(fit, fx$X_test)
  expect_snapshot_value(preds, style = "serialize", tolerance = 1e-10)
})

test_that("psvr mape / sym=+1 / poly — direct golden", {
  fx <- make_fixture()
  K  <- make_kernel("polynomial", degree = HP$degree, coef0 = HP$scale_factor)
  fit <- psvr(fx$X, fx$y, loss = "mape", sym = HP$a,
              kernel = K, C = HP$C, eps = HP$eps)
  preds <- predict(fit, fx$X_test)
  expect_snapshot_value(preds, style = "serialize", tolerance = 1e-10)
})

test_that("psvr mape / sym=+1 / linear — direct golden", {
  fx <- make_fixture()
  K  <- make_kernel("linear")
  fit <- psvr(fx$X, fx$y, loss = "mape", sym = HP$a,
              kernel = K, C = HP$C, eps = HP$eps)
  preds <- predict(fit, fx$X_test)
  expect_snapshot_value(preds, style = "serialize", tolerance = 1e-10)
})

test_that("psvr rmspe / no sym / RBF — direct golden", {
  fx <- make_fixture()
  K  <- make_kernel("rbf", sigma = HP$rbf_sigma)
  fit <- psvr(fx$X, fx$y, loss = "rmspe", kernel = K, gamma = HP$gamma)
  preds <- predict(fit, fx$X_test)
  expect_snapshot_value(preds, style = "serialize", tolerance = 1e-10)
})

test_that("psvr rmspe / no sym / poly — direct golden", {
  fx <- make_fixture()
  K  <- make_kernel("polynomial", degree = HP$degree, coef0 = HP$scale_factor)
  fit <- psvr(fx$X, fx$y, loss = "rmspe", kernel = K, gamma = HP$gamma)
  preds <- predict(fit, fx$X_test)
  expect_snapshot_value(preds, style = "serialize", tolerance = 1e-10)
})

test_that("psvr rmspe / no sym / linear — direct golden", {
  fx <- make_fixture()
  K  <- make_kernel("linear")
  fit <- psvr(fx$X, fx$y, loss = "rmspe", kernel = K, gamma = HP$gamma)
  preds <- predict(fit, fx$X_test)
  expect_snapshot_value(preds, style = "serialize", tolerance = 1e-10)
})

test_that("psvr rmspe / sym=+1 / RBF — direct golden", {
  fx <- make_fixture()
  K  <- make_kernel("rbf", sigma = HP$rbf_sigma)
  fit <- psvr(fx$X, fx$y, loss = "rmspe", sym = HP$a,
              kernel = K, gamma = HP$gamma)
  preds <- predict(fit, fx$X_test)
  expect_snapshot_value(preds, style = "serialize", tolerance = 1e-10)
})

test_that("psvr rmspe / sym=+1 / poly — direct golden", {
  fx <- make_fixture()
  K  <- make_kernel("polynomial", degree = HP$degree, coef0 = HP$scale_factor)
  fit <- psvr(fx$X, fx$y, loss = "rmspe", sym = HP$a,
              kernel = K, gamma = HP$gamma)
  preds <- predict(fit, fx$X_test)
  expect_snapshot_value(preds, style = "serialize", tolerance = 1e-10)
})

test_that("psvr rmspe / sym=+1 / linear — direct golden", {
  fx <- make_fixture()
  K  <- make_kernel("linear")
  fit <- psvr(fx$X, fx$y, loss = "rmspe", sym = HP$a,
              kernel = K, gamma = HP$gamma)
  preds <- predict(fit, fx$X_test)
  expect_snapshot_value(preds, style = "serialize", tolerance = 1e-10)
})

# ---- F5 field-presence contracts on psvr_fit -------------------------------

test_that("psvr_fit MAPE: alpha/alpha_star length N, beta length n_sv", {
  fx <- make_fixture()
  K  <- make_kernel("rbf", sigma = HP$rbf_sigma)
  fit <- suppressWarnings(
    psvr(fx$X, fx$y, loss = "mape", kernel = K, C = HP$C, eps = HP$eps)
  )
  N <- nrow(fx$X)
  expect_length(fit$alpha,      N)
  expect_length(fit$alpha_star, N)
  expect_equal(length(fit$beta), fit$n_sv)
  # On SV indices, beta equals alpha - alpha_star within numerical tol.
  beta_full <- fit$alpha - fit$alpha_star
  sv_idx <- which(abs(beta_full) > 1e-5)
  expect_equal(beta_full[sv_idx], fit$beta, tolerance = 1e-10)
})

test_that("psvr_fit RMSPE: alpha length N, alpha_star and beta are NULL", {
  fx <- make_fixture()
  K  <- make_kernel("rbf", sigma = HP$rbf_sigma)
  fit <- psvr(fx$X, fx$y, loss = "rmspe", kernel = K, gamma = HP$gamma)
  expect_length(fit$alpha, nrow(fx$X))
  expect_null(fit$alpha_star)
  expect_null(fit$beta)
})
