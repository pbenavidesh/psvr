# Tests for argument handling on the two public fitters. Deliberately a
# separate file: the suite has no other home for argument validation, and
# adding cases to test-psvr-direct.R would put them against a protected
# snapshot baseline.

set.seed(2026)
N <- 30L
X <- matrix(rnorm(N * 2L), N, 2L)
y <- rlnorm(N)
K <- make_kernel("rbf", sigma = 1)

test_that("psvr_mape() rejects unknown arguments passed through `...`", {
  expect_error(
    psvr_mape(X, y, kernel = K, C = 10, eps = 5, epsilon = 5),
    "`\\.\\.\\.` must be empty"
  )
})

test_that("psvr_mape() names every unknown argument, not just the first", {
  err <- expect_error(
    psvr_mape(X, y, kernel = K, C = 10, eps = 5,
              epsilon = 5, sigma = 2)
  )
  expect_match(conditionMessage(err), "epsilon")
  expect_match(conditionMessage(err), "sigma")
})

test_that("psvr_mape() still accepts the exact-match-only internal arguments", {
  # This is what `...` buys: alpha_couple sits after it and must remain
  # reachable by its exact name. Without the dots, `precomputed_Omega` would
  # be a partial-match prefix of `precomputed_Omega_s`. See ?psvr_mape.
  expect_no_error(
    psvr_mape(X, y, kernel = K, C = 10, eps = 5,
              alpha_couple = 0.5)
  )
})

test_that("psvr_rmspe() rejects unknown arguments too", {
  # psvr_rmspe() has no internal post-dots formals of its own, but it carries
  # `...` for the same reason psvr_mape() does: so a typo errors instead of
  # being silently ignored.
  expect_error(
    psvr_rmspe(X, y, kernel = K, gamma = 100, Gamma = 100),
    "`\\.\\.\\.` must be empty"
  )
})

test_that("the old `sym` argument partial-matches `sym_type` and errors", {
  # Migration path from the superseded psvr(). `sym_type` sits BEFORE `...`, so R
  # partial-matches `sym = 1L` onto it rather than routing it to the dots --
  # which means the diagnostic comes from match.arg(), not check_dots_empty().
  # Either way it errors rather than being silently ignored, and the message
  # names the argument that actually exists.
  expect_error(
    psvr_rmspe(X, y, kernel = K, gamma = 100, sym = 1L),
    "must be NULL or a character vector"
  )
  expect_error(
    psvr_mape(X, y, kernel = K, C = 10, eps = 5, sym = 1L),
    "must be NULL or a character vector"
  )
  # A string that is not one of the three levels gets the enumerated message.
  expect_error(
    psvr_mape(X, y, kernel = K, C = 10, eps = 5, sym_type = "symmetric"),
    "should be one of"
  )
})

test_that("psvr_mape() reports `reg` as unimplemented, not merely unknown", {
  # `reg` was a formal of the superseded psvr() whose only behaviour was to error
  # on any non-NULL value -- a not-implemented placeholder, not a validity
  # guard, so it was dropped rather than carried over. The targeted message
  # survives because check_dots_empty() alone would say the name is unknown
  # without saying that the feature does not exist yet.
  err <- expect_error(
    psvr_mape(X, y, kernel = K, C = 10, eps = 5, reg = 0.5)
  )
  expect_match(conditionMessage(err), "not implemented")
})
