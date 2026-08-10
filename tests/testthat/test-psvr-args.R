# Tests for psvr()'s argument handling. Deliberately a separate file: the
# suite has no other home for argument validation, and adding cases to
# test-psvr-direct.R would put them against a protected snapshot baseline.

set.seed(2026)
N <- 30L
X <- matrix(rnorm(N * 2L), N, 2L)
y <- rlnorm(N)
K <- make_kernel("rbf", sigma = 1)

test_that("psvr() rejects unknown arguments passed through `...`", {
  expect_error(
    psvr(X, y, loss = "mape", kernel = K, C = 10, eps = 5, epsilon = 5),
    "`\\.\\.\\.` must be empty"
  )
})

test_that("psvr() names every unknown argument, not just the first", {
  err <- expect_error(
    psvr(X, y, loss = "mape", kernel = K, C = 10, eps = 5,
         epsilon = 5, sigma = 2)
  )
  expect_match(conditionMessage(err), "epsilon")
  expect_match(conditionMessage(err), "sigma")
})

test_that("psvr() still accepts the exact-match-only internal arguments", {
  # This is what `...` buys: alpha_couple sits after it and must remain
  # reachable by its exact name. See the P-6a rationale in ?psvr.
  expect_no_error(
    psvr(X, y, loss = "mape", kernel = K, C = 10, eps = 5,
         alpha_couple = 0.5)
  )
})
