set.seed(404)
X_tr <- matrix(rnorm(30), 15, 2)
y_tr <- abs(rnorm(15)) + 1
K    <- make_kernel("rbf", sigma = 1)

# ── shape and class ──────────────────────────────────────────────────────────

test_that("mape_sym_svr returns psvr_mape_sym object with expected fields", {
  fit <- mape_sym_svr(X_tr, y_tr, kernel = K, C = 10, eps = 5, a = 1)
  expect_s3_class(fit, "psvr_mape_sym")
  expect_named(fit, c("beta", "b", "X_sv", "y_sv", "kernel",
                      "C", "eps", "a", "n_train", "p_train"))
  expect_true(is.numeric(fit$beta))
  expect_true(is.numeric(fit$b) && length(fit$b) == 1L)
  expect_equal(ncol(fit$X_sv), ncol(X_tr))
  expect_equal(nrow(fit$X_sv), length(fit$beta))
  expect_equal(length(fit$y_sv), length(fit$beta))
  expect_equal(fit$a, 1)
  expect_equal(fit$C, 10)
  expect_equal(fit$n_train, nrow(X_tr))
  expect_equal(fit$p_train, ncol(X_tr))
})

test_that("predict.psvr_mape_sym returns plain numeric vector of correct length", {
  fit   <- mape_sym_svr(X_tr, y_tr, kernel = K, C = 10, eps = 5, a = 1)
  X_new <- matrix(rnorm(10), 5, 2)
  preds <- predict(fit, X_new)
  expect_true(is.numeric(preds))
  expect_true(is.vector(preds))
  expect_length(preds, nrow(X_new))
})

test_that("predict.psvr_mape_sym returns plain vector for single-row newdata", {
  fit   <- mape_sym_svr(X_tr, y_tr, kernel = K, C = 10, eps = 5, a = 1)
  preds <- predict(fit, X_tr[1L, , drop = FALSE])
  expect_true(is.vector(preds))
  expect_length(preds, 1L)
})

# ── newdata column validation ────────────────────────────────────────────────

test_that("predict.psvr_mape_sym errors on column mismatch", {
  fit   <- mape_sym_svr(X_tr, y_tr, kernel = K, C = 10, eps = 5, a = 1)
  X_bad <- matrix(rnorm(20), 5, 4)
  expect_error(predict(fit, X_bad), "4 columns but model was trained on 2")
})

# ── input validation ─────────────────────────────────────────────────────────

test_that("mape_sym_svr errors when any y <= 0", {
  y_bad    <- y_tr
  y_bad[4] <- 0
  expect_error(mape_sym_svr(X_tr, y_bad, kernel = K, C = 10, eps = 5, a = 1),
               "strictly positive")
})

test_that("mape_sym_svr errors when C <= 0", {
  expect_error(mape_sym_svr(X_tr, y_tr, kernel = K, C =  0, eps = 5, a = 1), "`C`")
  expect_error(mape_sym_svr(X_tr, y_tr, kernel = K, C = -1, eps = 5, a = 1), "`C`")
})

test_that("mape_sym_svr errors when eps < 0", {
  expect_error(mape_sym_svr(X_tr, y_tr, kernel = K, C = 10, eps = -1, a = 1), "`eps`")
})

test_that("mape_sym_svr errors when a not in {-1, 1}", {
  expect_error(mape_sym_svr(X_tr, y_tr, kernel = K, C = 10, eps = 5, a =  0), "`a`")
  expect_error(mape_sym_svr(X_tr, y_tr, kernel = K, C = 10, eps = 5, a =  2), "`a`")
})

# ── box constraints: same bounds as Model 1 ──────────────────────────────────

test_that("mape_sym_svr dual variables satisfy box constraints", {
  C   <- 5
  fit <- mape_sym_svr(X_tr, y_tr, kernel = K, C = C, eps = 5, a = 1)
  upper <- 100 * C / fit$y_sv
  expect_true(all(abs(fit$beta) <= upper + 1e-4),
              info = paste("max violation:",
                           max(abs(fit$beta) - upper)))
})

# ── predictions are finite ───────────────────────────────────────────────────

test_that("mape_sym_svr predictions on training data are all finite", {
  fit   <- mape_sym_svr(X_tr, y_tr, kernel = K, C = 10, eps = 5, a = 1)
  preds <- predict(fit, X_tr)
  expect_true(all(is.finite(preds)))
})

# ── a = -1 variant runs without error ────────────────────────────────────────

test_that("mape_sym_svr works with a = -1 (odd symmetry)", {
  fit   <- mape_sym_svr(X_tr, y_tr, kernel = K, C = 10, eps = 5, a = -1)
  preds <- predict(fit, X_tr)
  expect_length(preds, nrow(X_tr))
  expect_true(all(is.finite(preds)))
})

# ── symmetry: Model 2 uses Ks = Ω + aΩ*, never ½Ω (distinct from Model 1) ──

test_that("mape_sym_svr and mape_svr give different predictions (Ks != Ω)", {
  fit1 <- mape_svr(    X_tr, y_tr, kernel = K, C = 10, eps = 5)
  fit2 <- mape_sym_svr(X_tr, y_tr, kernel = K, C = 10, eps = 5, a = 1)
  p1   <- predict(fit1, X_tr)
  p2   <- predict(fit2, X_tr)
  # The two models use different kernel matrices so predictions will differ
  expect_false(isTRUE(all.equal(p1, p2, tolerance = 1e-6)))
})

# ── print() ──────────────────────────────────────────────────────────────────

test_that("print.psvr_mape_sym produces output and returns object invisibly", {
  fit <- mape_sym_svr(X_tr, y_tr, kernel = K, C = 10, eps = 5, a = 1)
  out <- capture.output(ret <- print(fit))
  expect_true(length(out) > 0)
  expect_identical(ret, fit)
  expect_true(any(grepl("psvr_mape_sym", out)))
  expect_true(any(grepl("Symmetry", out)))
  expect_true(any(grepl("Support vectors", out)))
})

# ── coef() ───────────────────────────────────────────────────────────────────

test_that("coef.psvr_mape_sym returns named list with alpha, b, X_sv", {
  fit <- mape_sym_svr(X_tr, y_tr, kernel = K, C = 10, eps = 5, a = 1)
  co  <- coef(fit)
  expect_named(co, c("alpha", "b", "X_sv"))
  expect_identical(co$alpha, fit$beta)
  expect_identical(co$b,     fit$b)
  expect_identical(co$X_sv,  fit$X_sv)
})
