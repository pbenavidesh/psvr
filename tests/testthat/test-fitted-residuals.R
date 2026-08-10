# fitted() / residuals() for psvr_fit and the four legacy classes.

make_data <- function(N = 40L, p = 2L, seed = 11L) {
  set.seed(seed)
  X <- matrix(runif(N * p, 0.5, 3), N, p)
  list(X = X, y = 2 + X[, 1]^2 + 0.5 * X[, 2] + abs(rnorm(N, 0, 0.15)))
}

# The four models as psvr() calls, keyed by label.
fit_all <- function(d, K = make_kernel("rbf", sigma = 0.8)) {
  suppressWarnings(list(
    M1 = psvr(d$X, d$y, loss = "mape",  kernel = K, C = 1, eps = 0.1),
    M2 = psvr(d$X, d$y, loss = "mape",  kernel = K, C = 1, eps = 0.1, sym = 1L),
    M3 = psvr(d$X, d$y, loss = "rmspe", kernel = K, gamma = 10),
    M4 = psvr(d$X, d$y, loss = "rmspe", kernel = K, gamma = 10, sym = 1L)
  ))
}

TYPES <- c("response", "percentage", "multiplicative")

test_that("residuals() returns length N for all four models x three types", {
  d <- make_data()
  N <- length(d$y)
  for (nm in names(fit_all(d))) {
    f <- fit_all(d)[[nm]]
    for (ty in TYPES) {
      expect_length(residuals(f, type = ty), N)
    }
    expect_length(fitted(f), N)
  }
})

test_that("response == percentage * y for all four models", {
  d <- make_data()
  for (nm in names(fit_all(d))) {
    f <- fit_all(d)[[nm]]
    expect_equal(residuals(f, "response"),
                 residuals(f, "percentage") * d$y,
                 tolerance = 1e-12,
                 info = nm)
  }
})

test_that("response == multiplicative * fitted() for all four models", {
  d <- make_data()
  for (nm in names(fit_all(d))) {
    f <- fit_all(d)[[nm]]
    expect_equal(residuals(f, "response"),
                 residuals(f, "multiplicative") * fitted(f),
                 tolerance = 1e-12,
                 info = nm)
  }
})

test_that("residuals() defaults to type = 'response'", {
  d <- make_data()
  f <- fit_all(d)$M3
  expect_identical(residuals(f), residuals(f, type = "response"))
})

test_that("the three types are genuinely different quantities", {
  # Guards against a denominator mix-up: percentage divides by y,
  # multiplicative by yhat, and they must not collapse into each other.
  d <- make_data()
  f <- fit_all(d)$M3
  expect_false(isTRUE(all.equal(residuals(f, "percentage"),
                                residuals(f, "multiplicative"))))
  expect_equal(residuals(f, "percentage"), (d$y - fitted(f)) / d$y,
               tolerance = 1e-12)
  expect_equal(residuals(f, "multiplicative"), (d$y - fitted(f)) / fitted(f),
               tolerance = 1e-12)
})

test_that("fitted() agrees with predict(object, X_train) for all four models", {
  # Validates the stored values against the independent predict() path,
  # which rebuilds the kernel from X. This is the load-bearing test: the
  # fitted values are recovered from solver state, never recomputed.
  d <- make_data()
  for (nm in names(fit_all(d))) {
    f  <- fit_all(d)[[nm]]
    pr <- predict(f, d$X)
    expect_equal(fitted(f), pr, tolerance = 1e-10, info = nm)
    expect_lt(max(abs(fitted(f) - pr) / pmax(abs(pr), 1e-300)), 1e-10)
  }
})

test_that("fitted()/predict() agreement holds under the LS-SVR preconditioner", {
  # The KKT identity used for the LS-SVR fitted values is derived separately
  # for the preconditioned branch; exercise both.
  set.seed(3L)
  X <- matrix(runif(60, 0.5, 3), 30, 2)
  y <- exp(rnorm(30, 1, 1.5))                    # max(y)/min(y) >> 10
  K <- make_kernel("rbf", sigma = 0.8)
  for (pc in c("always", "never")) {
    for (s in list(NULL, 1L)) {
      f <- psvr(X, y, loss = "rmspe", kernel = K, gamma = 10,
                sym = s, precondition = pc)
      expect_true(identical(f$solver_meta$precondition_applied, pc == "always"))
      expect_equal(fitted(f), predict(f, X), tolerance = 1e-9)
    }
  }
})

test_that("fitted() agrees with predict() across kernels", {
  d <- make_data()
  for (K in list(make_kernel("rbf", sigma = 0.8),
                 make_kernel("linear"),
                 make_kernel("polynomial", degree = 2L, coef0 = 1))) {
    f <- psvr(d$X, d$y, loss = "rmspe", kernel = K, gamma = 10)
    expect_equal(fitted(f), predict(f, d$X), tolerance = 1e-9)
  }
})

test_that("training MAPE from percentage residuals matches a manual computation", {
  d <- make_data()
  f <- fit_all(d)$M3
  expect_equal(mean(abs(residuals(f, "percentage"))) * 100,
               mean(abs((d$y - fitted(f)) / d$y)) * 100,
               tolerance = 1e-12)
})

test_that("near-zero fitted values warn once and return length N", {
  # Force a fitted value onto the near-zero floor by editing the stored
  # vector: the warning path depends only on fitted_values, and provoking it
  # through a real fit would require a pathological hyperparameter set.
  d <- make_data()
  f <- fit_all(d)$M3
  N <- length(d$y)

  f_bad <- f
  f_bad$fitted_values[c(2L, 5L)] <- c(0, -1e-30)

  expect_warning(r <- residuals(f_bad, "multiplicative"), "2 of 40")
  expect_length(r, N)
  expect_false(is.finite(r[2L]))          # divide by exactly zero
  # Not dropped: every other entry survives untouched and stays aligned.
  expect_equal(r[-c(2L, 5L)],
               residuals(f, "multiplicative")[-c(2L, 5L)],
               tolerance = 1e-12)

  # Singular form, and only one warning for many bad values.
  f_one <- f
  f_one$fitted_values[3L] <- 0
  expect_warning(residuals(f_one, "multiplicative"), "1 of 40 fitted value is")

  # The other two types never warn — y > 0 is a fit-time invariant.
  expect_silent(residuals(f_bad, "response"))
  expect_silent(residuals(f_bad, "percentage"))
})

test_that("no warning fires on a healthy fit", {
  d <- make_data()
  for (nm in names(fit_all(d))) {
    expect_silent(residuals(fit_all(d)[[nm]], "multiplicative"))
  }
})

test_that("the four legacy classes support fitted() and residuals()", {
  d <- make_data()
  N <- length(d$y)
  K <- make_kernel("rbf", sigma = 0.8)
  # The internal fitters, not psvr(): these four classes are what the parsnip
  # engine fit wrappers return, and psvr() returns psvr_fit instead.
  legacy <- suppressWarnings(list(
    psvr_mape      = psvr:::.fit_mape(d$X, d$y, K, C = 1, eps = 0.1),
    psvr_mape_sym  = psvr:::.fit_mape_sym(d$X, d$y, K, C = 1, eps = 0.1, a = 1L),
    psvr_rmspe     = psvr:::.fit_rmspe(d$X, d$y, K, gamma = 10),
    psvr_rmspe_sym = psvr:::.fit_rmspe_sym(d$X, d$y, K, gamma = 10, a = 1L)
  ))
  for (nm in names(legacy)) {
    f <- legacy[[nm]]
    expect_s3_class(f, nm)
    expect_length(fitted(f), N)
    expect_equal(fitted(f), predict(f, d$X), tolerance = 1e-10, info = nm)
    for (ty in TYPES) expect_length(residuals(f, type = ty), N)
    expect_equal(residuals(f, "response"), residuals(f, "percentage") * d$y,
                 tolerance = 1e-12, info = nm)
  }
})

test_that("objects lacking the new fields give an actionable error", {
  d <- make_data()
  f <- fit_all(d)$M3

  stale <- f
  stale$fitted_values <- NULL
  stale$y_train       <- NULL

  expect_error(fitted(stale),    "psvr < 0\\.0\\.2\\.9010")
  expect_error(residuals(stale), "psvr < 0\\.0\\.2\\.9010")
  expect_error(fitted(stale),    "Refit the model")
  # Half-populated objects are caught too.
  half <- f; half$y_train <- NULL
  expect_error(residuals(half), "Refit the model")
})

test_that("an unknown residual type is rejected", {
  d <- make_data()
  expect_error(residuals(fit_all(d)$M3, type = "pct"))
})
