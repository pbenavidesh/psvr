## Tests for the deprecated public rmspe_lssvr() wrapper. The wrapper
## emits .Deprecated("psvr") on every call; the contract is asserted
## explicitly in the first ("shape and class") test, and other tests use
## a quiet helper to keep the test output focused on behavior.

set.seed(101)
X_tr <- matrix(rnorm(30), 15, 2)
y_tr <- abs(rnorm(15)) + 1     # strictly positive
K    <- make_kernel("rbf", sigma = 1)

.q_rmspe_lssvr <- function(...) suppressWarnings(rmspe_lssvr(...))

# ── shape, class, and deprecation contract ───────────────────────────────────

test_that("rmspe_lssvr emits deprecation notice and returns legacy psvr_rmspe shape", {
  expect_warning(
    fit <- rmspe_lssvr(X_tr, y_tr, kernel = K, gamma = 1),
    regexp = "deprecated|psvr"
  )
  expect_s3_class(fit, "psvr_rmspe")
  expect_named(fit, c("alpha", "b", "X_train", "kernel",
                      "gamma", "n_train", "p_train",
                      "precondition_applied"))
  expect_length(fit$alpha, nrow(X_tr))
  expect_true(is.numeric(fit$b) && length(fit$b) == 1L)
  expect_equal(fit$gamma, 1)
  expect_equal(fit$n_train, nrow(X_tr))
  expect_equal(fit$p_train, ncol(X_tr))
})

test_that("predict.psvr_rmspe returns plain numeric vector of correct length", {
  fit   <- .q_rmspe_lssvr(X_tr, y_tr, kernel = K, gamma = 1)
  X_new <- matrix(rnorm(10), 5, 2)
  preds <- predict(fit, X_new)
  expect_true(is.numeric(preds))
  expect_true(is.vector(preds))
  expect_length(preds, nrow(X_new))
})

test_that("predict.psvr_rmspe returns plain vector for single-row newdata", {
  fit   <- .q_rmspe_lssvr(X_tr, y_tr, kernel = K, gamma = 1)
  preds <- predict(fit, X_tr[1L, , drop = FALSE])
  expect_true(is.vector(preds))
  expect_length(preds, 1L)
})

# ── newdata column validation ────────────────────────────────────────────────

test_that("predict.psvr_rmspe errors on column mismatch", {
  fit   <- .q_rmspe_lssvr(X_tr, y_tr, kernel = K, gamma = 1)
  X_bad <- matrix(rnorm(5), 5, 1)
  expect_error(predict(fit, X_bad), "1 column but model was trained on 2")
})

# ── input validation ─────────────────────────────────────────────────────────

test_that("rmspe_lssvr errors when any y <= 0", {
  y_bad <- y_tr
  y_bad[3] <- -0.5
  expect_error(.q_rmspe_lssvr(X_tr, y_bad, kernel = K, gamma = 1),
               "strictly positive")
})

test_that("rmspe_lssvr errors when any y == 0", {
  y_bad <- y_tr
  y_bad[1] <- 0
  expect_error(.q_rmspe_lssvr(X_tr, y_bad, kernel = K, gamma = 1),
               "strictly positive")
})

test_that("rmspe_lssvr errors when gamma <= 0", {
  expect_error(.q_rmspe_lssvr(X_tr, y_tr, kernel = K, gamma =  0), "`gamma`")
  expect_error(.q_rmspe_lssvr(X_tr, y_tr, kernel = K, gamma = -1), "`gamma`")
})

# ── training fit: linear system equation f(xk) + ek = yk ───────────────────
#
# KKT gives ek = (yk²/Γ)·αk, so predict(fit, X_tr) + (y²/Γ)·α = y exactly.
# This holds for any Γ and is the direct check of the system solution.

test_that("rmspe_lssvr solution satisfies the KKT equality f(xk) + ek = yk", {
  gamma <- 2
  fit   <- .q_rmspe_lssvr(X_tr, y_tr, kernel = K, gamma = gamma)
  preds <- predict(fit, X_tr)
  ek    <- (y_tr^2 / gamma) * fit$alpha   # ek = (yk²/Γ)·αk from KKT
  expect_equal(preds + ek, y_tr, tolerance = 1e-6)
})

# ── KKT consistency: Σ αk = 0 ────────────────────────────────────────────────

test_that("rmspe_lssvr solution satisfies balance condition sum(alpha) == 0", {
  fit <- .q_rmspe_lssvr(X_tr, y_tr, kernel = K, gamma = 1)
  expect_equal(sum(fit$alpha), 0, tolerance = 1e-10)
})

# ── print() ──────────────────────────────────────────────────────────────────

test_that("print.psvr_rmspe produces output and returns object invisibly", {
  fit <- .q_rmspe_lssvr(X_tr, y_tr, kernel = K, gamma = 1)
  out <- capture.output(ret <- print(fit))
  expect_true(length(out) > 0)
  expect_identical(ret, fit)
  expect_true(any(grepl("psvr_rmspe", out)))
  expect_true(any(grepl("Gamma", out)))
})

# ── coef() ───────────────────────────────────────────────────────────────────

test_that("coef.psvr_rmspe returns named list with alpha, b, X_sv", {
  fit <- .q_rmspe_lssvr(X_tr, y_tr, kernel = K, gamma = 1)
  co  <- coef(fit)
  expect_named(co, c("alpha", "b", "X_sv"))
  expect_identical(co$alpha, fit$alpha)
  expect_identical(co$b,     fit$b)
  expect_identical(co$X_sv,  fit$X_train)
})
