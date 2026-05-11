## Tests for the deprecated public mape_svr() wrapper. The wrapper itself
## emits .Deprecated("psvr") on every call; the contract is asserted
## explicitly in the first ("shape and class") test, and other tests use
## a quiet helper to keep the test output focused on behavior.

set.seed(303)
X_tr <- matrix(rnorm(30), 15, 2)
y_tr <- abs(rnorm(15)) + 1
K    <- make_kernel("rbf", sigma = 1)

# Quiet wrapper for behavior tests only — the dedicated contract test
# below asserts the .Deprecated() notice explicitly.
.q_mape_svr <- function(...) suppressWarnings(mape_svr(...))

# ── shape, class, and deprecation contract ───────────────────────────────────

test_that("mape_svr emits deprecation notice and returns legacy psvr_mape shape", {
  expect_warning(
    fit <- mape_svr(X_tr, y_tr, kernel = K, C = 10, eps = 5),
    regexp = "deprecated|psvr"
  )
  expect_s3_class(fit, "psvr_mape")
  expect_named(fit, c("beta", "alpha", "alpha_star", "b",
                      "X_sv", "y_sv", "kernel", "C", "eps",
                      "n_train", "p_train", "iterations", "converged",
                      "block_k4"))
  expect_true(is.numeric(fit$beta))
  expect_true(is.numeric(fit$b) && length(fit$b) == 1L)
  expect_equal(ncol(fit$X_sv), ncol(X_tr))
  expect_equal(nrow(fit$X_sv), length(fit$beta))
  expect_equal(length(fit$y_sv), length(fit$beta))
  expect_equal(fit$C, 10)
  expect_equal(fit$n_train, nrow(X_tr))
  expect_equal(fit$p_train, ncol(X_tr))
  # F5: full-length pre-pruning duals exposed for warm-start.
  expect_length(fit$alpha,      nrow(X_tr))
  expect_length(fit$alpha_star, nrow(X_tr))
  # F5: SMO meta propagated through the fitter.
  expect_true(is.numeric(fit$iterations) || is.integer(fit$iterations))
  expect_true(is.logical(fit$converged))
})

test_that("predict.psvr_mape returns plain numeric vector of correct length", {
  fit   <- .q_mape_svr(X_tr, y_tr, kernel = K, C = 10, eps = 5)
  X_new <- matrix(rnorm(10), 5, 2)
  preds <- predict(fit, X_new)
  expect_true(is.numeric(preds))
  expect_true(is.vector(preds))
  expect_length(preds, nrow(X_new))
})

test_that("predict.psvr_mape returns plain vector for single-row newdata", {
  fit   <- .q_mape_svr(X_tr, y_tr, kernel = K, C = 10, eps = 5)
  preds <- predict(fit, X_tr[1L, , drop = FALSE])
  expect_true(is.vector(preds))
  expect_length(preds, 1L)
})

# ── newdata column validation ────────────────────────────────────────────────

test_that("predict.psvr_mape errors on column mismatch", {
  fit   <- .q_mape_svr(X_tr, y_tr, kernel = K, C = 10, eps = 5)
  X_bad <- matrix(rnorm(15), 5, 3)
  expect_error(predict(fit, X_bad), "3 columns but model was trained on 2")
})

# ── input validation ─────────────────────────────────────────────────────────

test_that("mape_svr errors when any y <= 0", {
  y_bad    <- y_tr
  y_bad[5] <- -1
  expect_error(.q_mape_svr(X_tr, y_bad, kernel = K, C = 10, eps = 5),
               "strictly positive")
})

test_that("mape_svr errors when any y == 0", {
  y_bad    <- y_tr
  y_bad[1] <- 0
  expect_error(.q_mape_svr(X_tr, y_bad, kernel = K, C = 10, eps = 5),
               "strictly positive")
})

test_that("mape_svr errors when C <= 0", {
  expect_error(.q_mape_svr(X_tr, y_tr, kernel = K, C =  0, eps = 5), "`C`")
  expect_error(.q_mape_svr(X_tr, y_tr, kernel = K, C = -1, eps = 5), "`C`")
})

test_that("mape_svr errors when eps < 0", {
  expect_error(.q_mape_svr(X_tr, y_tr, kernel = K, C = 10, eps = -1), "`eps`")
})

# ── box constraints: |βk| ≤ 100C/yk for all support vectors ─────────────────

test_that("mape_svr dual variables satisfy box constraints", {
  C   <- 5
  fit <- .q_mape_svr(X_tr, y_tr, kernel = K, C = C, eps = 5)

  # Per-sample upper bound on |β| at each support vector
  upper <- 100 * C / fit$y_sv

  # Allow a small numerical tolerance from the QP solver
  expect_true(all(abs(fit$beta) <= upper + 1e-4),
              info = paste("max violation:",
                           max(abs(fit$beta) - upper)))
})

# ── predictions are finite ───────────────────────────────────────────────────

test_that("mape_svr predictions on training data are all finite", {
  fit   <- .q_mape_svr(X_tr, y_tr, kernel = K, C = 10, eps = 5)
  preds <- predict(fit, X_tr)
  expect_true(all(is.finite(preds)))
})

# ── eps = 0 gives a tighter fit ──────────────────────────────────────────────

test_that("mape_svr with eps = 0 fits training data more tightly than eps = 10", {
  fit0  <- .q_mape_svr(X_tr, y_tr, kernel = K, C = 50, eps =  0)
  fit10 <- .q_mape_svr(X_tr, y_tr, kernel = K, C = 50, eps = 10)

  mape0  <- mean(abs(predict(fit0,  X_tr) - y_tr) / y_tr)
  mape10 <- mean(abs(predict(fit10, X_tr) - y_tr) / y_tr)

  expect_lt(mape0, mape10 + 1e-6)
})

# ── print() ──────────────────────────────────────────────────────────────────

test_that("print.psvr_mape produces output and returns object invisibly", {
  fit <- .q_mape_svr(X_tr, y_tr, kernel = K, C = 10, eps = 5)
  out <- capture.output(ret <- print(fit))
  expect_true(length(out) > 0)
  expect_identical(ret, fit)
  expect_true(any(grepl("psvr_mape", out)))
  expect_true(any(grepl("C", out)))
  expect_true(any(grepl("Support vectors", out)))
})

# ── coef() ───────────────────────────────────────────────────────────────────

test_that("coef.psvr_mape returns named list with alpha, b, X_sv", {
  fit <- .q_mape_svr(X_tr, y_tr, kernel = K, C = 10, eps = 5)
  co  <- coef(fit)
  expect_named(co, c("alpha", "b", "X_sv"))
  expect_identical(co$alpha, fit$beta)
  expect_identical(co$b,     fit$b)
  expect_identical(co$X_sv,  fit$X_sv)
})
