# Tests for the symmetric rescaling preconditioner (Remark 17) used by the
# RMSPE LS-SVR fitters. Covers both rmspe_lssvr() and rmspe_sym_lssvr().
#
# The preconditioner is a strict change of variable: in exact arithmetic,
# precondition = "always" and precondition = "never" produce identical
# alpha, b, and predictions. These tests verify that property to within
# machine epsilon for moderate rho, and to a looser numerical-stability
# tolerance at extreme rho.

# ── helpers ──────────────────────────────────────────────────────────────────

make_synth <- function(rho, N = 60, seed = 1L) {
  set.seed(seed)
  x <- seq(0, 1, length.out = N)
  y_clean <- 1 + (rho - 1) * x^2          # y_clean in [1, rho]
  y <- y_clean + stats::rnorm(N, sd = 0.01 * rho)
  y <- pmax(y, 1e-3)                      # keep strictly positive
  list(
    X = matrix(x, ncol = 1),
    y = y
  )
}

K_rbf <- make_kernel("rbf", sigma = 0.3)

# ── 1. precondition = "never" reproduces existing behaviour ──────────────────

test_that("rmspe_lssvr precondition='never' satisfies legacy KKT equality", {
  d     <- make_synth(rho = 8, N = 30, seed = 11L)
  gamma <- 2
  fit   <- rmspe_lssvr(d$X, d$y, kernel = K_rbf, gamma = gamma,
                       precondition = "never")
  expect_false(fit$precondition_applied)

  # Legacy KKT identity: f(x_k) + e_k = y_k with e_k = (y_k^2/Γ)·α_k.
  preds <- predict(fit, d$X)
  ek    <- (d$y^2 / gamma) * fit$alpha
  expect_equal(preds + ek, d$y, tolerance = 1e-6)
  expect_equal(sum(fit$alpha), 0, tolerance = 1e-10)
})

test_that("rmspe_sym_lssvr precondition='never' satisfies legacy KKT equality", {
  d     <- make_synth(rho = 8, N = 30, seed = 12L)
  gamma <- 2
  fit   <- rmspe_sym_lssvr(d$X, d$y, kernel = K_rbf, gamma = gamma, a = 1,
                           precondition = "never")
  expect_false(fit$precondition_applied)

  preds <- predict(fit, d$X)
  ek    <- (d$y^2 / gamma) * fit$alpha
  expect_equal(preds + ek, d$y, tolerance = 1e-6)
  expect_equal(sum(fit$alpha), 0, tolerance = 1e-10)
})

# ── 2. precondition = "always" matches "never" to machine epsilon ───────────

test_that("rmspe_lssvr precondition='always' == 'never' on moderate-rho data", {
  d <- make_synth(rho = 10, N = 40, seed = 21L)
  fit_n <- rmspe_lssvr(d$X, d$y, kernel = K_rbf, gamma = 1,
                       precondition = "never")
  fit_a <- rmspe_lssvr(d$X, d$y, kernel = K_rbf, gamma = 1,
                       precondition = "always")
  expect_true(fit_a$precondition_applied)
  expect_false(fit_n$precondition_applied)

  X_te  <- matrix(seq(0, 1, length.out = 25), ncol = 1)
  expect_lt(max(abs(fit_n$alpha - fit_a$alpha)),                     1e-8)
  expect_lt(abs(fit_n$b - fit_a$b),                                  1e-8)
  expect_lt(max(abs(predict(fit_n, X_te) - predict(fit_a, X_te))),   1e-8)
})

test_that("rmspe_sym_lssvr precondition='always' == 'never' on moderate-rho data", {
  d <- make_synth(rho = 10, N = 40, seed = 22L)
  fit_n <- rmspe_sym_lssvr(d$X, d$y, kernel = K_rbf, gamma = 1, a = 1,
                           precondition = "never")
  fit_a <- rmspe_sym_lssvr(d$X, d$y, kernel = K_rbf, gamma = 1, a = 1,
                           precondition = "always")
  expect_true(fit_a$precondition_applied)

  X_te <- matrix(seq(0, 1, length.out = 25), ncol = 1)
  expect_lt(max(abs(fit_n$alpha - fit_a$alpha)),                     1e-8)
  expect_lt(abs(fit_n$b - fit_a$b),                                  1e-8)
  expect_lt(max(abs(predict(fit_n, X_te) - predict(fit_a, X_te))),   1e-8)
})

# ── 3. precondition = "always" stays numerically stable at extreme rho ──────

test_that("rmspe_lssvr precondition='always' is numerically stable at rho=1000", {
  d <- make_synth(rho = 1000, N = 60, seed = 31L)
  fit_n <- rmspe_lssvr(d$X, d$y, kernel = K_rbf, gamma = 1,
                       precondition = "never")
  fit_a <- rmspe_lssvr(d$X, d$y, kernel = K_rbf, gamma = 1,
                       precondition = "always")

  X_te <- matrix(seq(0, 1, length.out = 25), ncol = 1)
  expect_lt(abs(fit_n$b - fit_a$b),                                  1e-6)
  expect_lt(max(abs(predict(fit_n, X_te) - predict(fit_a, X_te))),   1e-6)
})

test_that("rmspe_sym_lssvr precondition='always' is numerically stable at rho=1000", {
  d <- make_synth(rho = 1000, N = 60, seed = 32L)
  fit_n <- rmspe_sym_lssvr(d$X, d$y, kernel = K_rbf, gamma = 1, a = 1,
                           precondition = "never")
  fit_a <- rmspe_sym_lssvr(d$X, d$y, kernel = K_rbf, gamma = 1, a = 1,
                           precondition = "always")

  X_te <- matrix(seq(0, 1, length.out = 25), ncol = 1)
  expect_lt(abs(fit_n$b - fit_a$b),                                  1e-6)
  expect_lt(max(abs(predict(fit_n, X_te) - predict(fit_a, X_te))),   1e-6)
})

# ── 4. precondition = "auto" activates at the correct threshold ─────────────

test_that("rmspe_lssvr precondition='auto' off below default threshold", {
  d   <- make_synth(rho = 5, N = 30, seed = 41L)
  fit_auto <- rmspe_lssvr(d$X, d$y, kernel = K_rbf, gamma = 1,
                          precondition = "auto")
  fit_n    <- rmspe_lssvr(d$X, d$y, kernel = K_rbf, gamma = 1,
                          precondition = "never")
  expect_false(fit_auto$precondition_applied)

  X_te <- matrix(seq(0, 1, length.out = 25), ncol = 1)
  expect_lt(max(abs(predict(fit_auto, X_te) - predict(fit_n, X_te))), 1e-8)
})

test_that("rmspe_lssvr precondition='auto' on above default threshold", {
  d   <- make_synth(rho = 20, N = 30, seed = 42L)
  fit_auto <- rmspe_lssvr(d$X, d$y, kernel = K_rbf, gamma = 1,
                          precondition = "auto")
  fit_n    <- rmspe_lssvr(d$X, d$y, kernel = K_rbf, gamma = 1,
                          precondition = "never")
  expect_true(fit_auto$precondition_applied)

  X_te <- matrix(seq(0, 1, length.out = 25), ncol = 1)
  expect_lt(max(abs(predict(fit_auto, X_te) - predict(fit_n, X_te))), 1e-8)
})

test_that("rmspe_lssvr precondition=<numeric> uses the supplied threshold", {
  d        <- make_synth(rho = 8, N = 30, seed = 43L)
  fit_off  <- rmspe_lssvr(d$X, d$y, kernel = K_rbf, gamma = 1,
                          precondition = 50)
  fit_on   <- rmspe_lssvr(d$X, d$y, kernel = K_rbf, gamma = 1,
                          precondition = 5)
  expect_false(fit_off$precondition_applied)
  expect_true(fit_on$precondition_applied)
})

test_that("rmspe_sym_lssvr precondition='auto' off below default threshold", {
  d   <- make_synth(rho = 5, N = 30, seed = 44L)
  fit_auto <- rmspe_sym_lssvr(d$X, d$y, kernel = K_rbf, gamma = 1, a = 1,
                              precondition = "auto")
  fit_n    <- rmspe_sym_lssvr(d$X, d$y, kernel = K_rbf, gamma = 1, a = 1,
                              precondition = "never")
  expect_false(fit_auto$precondition_applied)

  X_te <- matrix(seq(0, 1, length.out = 25), ncol = 1)
  expect_lt(max(abs(predict(fit_auto, X_te) - predict(fit_n, X_te))), 1e-8)
})

test_that("rmspe_sym_lssvr precondition='auto' on above default threshold", {
  d   <- make_synth(rho = 20, N = 30, seed = 45L)
  fit_auto <- rmspe_sym_lssvr(d$X, d$y, kernel = K_rbf, gamma = 1, a = 1,
                              precondition = "auto")
  fit_n    <- rmspe_sym_lssvr(d$X, d$y, kernel = K_rbf, gamma = 1, a = 1,
                              precondition = "never")
  expect_true(fit_auto$precondition_applied)

  X_te <- matrix(seq(0, 1, length.out = 25), ncol = 1)
  expect_lt(max(abs(predict(fit_auto, X_te) - predict(fit_n, X_te))), 1e-8)
})

test_that("rmspe_sym_lssvr precondition=<numeric> uses the supplied threshold", {
  d        <- make_synth(rho = 8, N = 30, seed = 46L)
  fit_off  <- rmspe_sym_lssvr(d$X, d$y, kernel = K_rbf, gamma = 1, a = 1,
                              precondition = 50)
  fit_on   <- rmspe_sym_lssvr(d$X, d$y, kernel = K_rbf, gamma = 1, a = 1,
                              precondition = 5)
  expect_false(fit_off$precondition_applied)
  expect_true(fit_on$precondition_applied)
})

# ── 5. precondition argument validation ─────────────────────────────────────

test_that("rmspe_lssvr rejects invalid `precondition` values", {
  d <- make_synth(rho = 5, N = 20, seed = 51L)

  expect_error(
    rmspe_lssvr(d$X, d$y, kernel = K_rbf, gamma = 1, precondition = "yes"),
    "precondition"
  )
  expect_error(
    rmspe_lssvr(d$X, d$y, kernel = K_rbf, gamma = 1, precondition = "true"),
    "precondition"
  )
  expect_error(
    rmspe_lssvr(d$X, d$y, kernel = K_rbf, gamma = 1, precondition = -1),
    "precondition"
  )
  expect_error(
    rmspe_lssvr(d$X, d$y, kernel = K_rbf, gamma = 1, precondition = 0),
    "precondition"
  )
  expect_error(
    rmspe_lssvr(d$X, d$y, kernel = K_rbf, gamma = 1, precondition = c(5, 10)),
    "precondition"
  )
  expect_error(
    rmspe_lssvr(d$X, d$y, kernel = K_rbf, gamma = 1, precondition = NA),
    "precondition"
  )
  expect_error(
    rmspe_lssvr(d$X, d$y, kernel = K_rbf, gamma = 1, precondition = TRUE),
    "precondition"
  )
})

test_that("rmspe_sym_lssvr rejects invalid `precondition` values", {
  d <- make_synth(rho = 5, N = 20, seed = 52L)

  expect_error(
    rmspe_sym_lssvr(d$X, d$y, kernel = K_rbf, gamma = 1, a = 1,
                    precondition = "yes"),
    "precondition"
  )
  expect_error(
    rmspe_sym_lssvr(d$X, d$y, kernel = K_rbf, gamma = 1, a = 1,
                    precondition = -1),
    "precondition"
  )
  expect_error(
    rmspe_sym_lssvr(d$X, d$y, kernel = K_rbf, gamma = 1, a = 1,
                    precondition = c(5, 10)),
    "precondition"
  )
})
