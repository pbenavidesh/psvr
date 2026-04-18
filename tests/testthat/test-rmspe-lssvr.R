set.seed(101)
X_tr <- matrix(rnorm(30), 15, 2)
y_tr <- abs(rnorm(15)) + 1     # strictly positive
K    <- make_kernel("rbf", sigma = 1)

# ── shape and class ──────────────────────────────────────────────────────────

test_that("rmspe_lssvr returns psvr_rmspe object with expected fields", {
  fit <- rmspe_lssvr(X_tr, y_tr, kernel = K, gamma = 1)
  expect_s3_class(fit, "psvr_rmspe")
  expect_named(fit, c("alpha", "b", "X_train", "kernel"))
  expect_length(fit$alpha, nrow(X_tr))
  expect_true(is.numeric(fit$b) && length(fit$b) == 1L)
})

test_that("predict.psvr_rmspe returns vector of correct length", {
  fit   <- rmspe_lssvr(X_tr, y_tr, kernel = K, gamma = 1)
  X_new <- matrix(rnorm(10), 5, 2)
  preds <- predict(fit, X_new)
  expect_true(is.numeric(preds))
  expect_length(preds, nrow(X_new))
})

# ── input validation ─────────────────────────────────────────────────────────

test_that("rmspe_lssvr errors when any y <= 0", {
  y_bad <- y_tr
  y_bad[3] <- -0.5
  expect_error(rmspe_lssvr(X_tr, y_bad, kernel = K, gamma = 1),
               "strictly positive")
})

test_that("rmspe_lssvr errors when any y == 0", {
  y_bad <- y_tr
  y_bad[1] <- 0
  expect_error(rmspe_lssvr(X_tr, y_bad, kernel = K, gamma = 1),
               "strictly positive")
})

test_that("rmspe_lssvr errors when gamma <= 0", {
  expect_error(rmspe_lssvr(X_tr, y_tr, kernel = K, gamma =  0), "`gamma`")
  expect_error(rmspe_lssvr(X_tr, y_tr, kernel = K, gamma = -1), "`gamma`")
})

# ── training fit: linear system equation f(xk) + ek = yk ───────────────────
#
# KKT gives ek = (yk²/Γ)·αk, so predict(fit, X_tr) + (y²/Γ)·α = y exactly.
# This holds for any Γ and is the direct check of the system solution.

test_that("rmspe_lssvr solution satisfies the KKT equality f(xk) + ek = yk", {
  gamma <- 2
  fit   <- rmspe_lssvr(X_tr, y_tr, kernel = K, gamma = gamma)
  preds <- predict(fit, X_tr)
  ek    <- (y_tr^2 / gamma) * fit$alpha   # ek = (yk²/Γ)·αk from KKT
  expect_equal(preds + ek, y_tr, tolerance = 1e-8)
})

# ── KKT consistency: Σ αk = 0 ────────────────────────────────────────────────

test_that("rmspe_lssvr solution satisfies balance condition sum(alpha) == 0", {
  fit <- rmspe_lssvr(X_tr, y_tr, kernel = K, gamma = 1)
  expect_equal(sum(fit$alpha), 0, tolerance = 1e-10)
})
