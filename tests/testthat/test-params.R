test_that("margin_percentage() returns a dials param with range [1, 20]", {
  p <- margin_percentage()
  expect_s3_class(p, "quant_param")
  r <- dials::range_get(p, original = FALSE)
  expect_equal(r$lower, 1)
  expect_equal(r$upper, 20)
})

test_that("margin_percentage() accepts a custom range", {
  p <- margin_percentage(range = c(0.5, 10))
  r <- dials::range_get(p, original = FALSE)
  expect_equal(r$lower, 0.5)
  expect_equal(r$upper, 10)
})

test_that("sigma_heuristic() returns a positive scalar on a small matrix", {
  set.seed(42)
  X <- matrix(rnorm(200), ncol = 4)
  result <- sigma_heuristic(X)
  expect_length(result, 1L)
  expect_true(is.numeric(result))
  expect_gt(result, 0)
})

test_that("sigma_heuristic() subsamples when nrow(X) > sample_size", {
  set.seed(1)
  X <- matrix(rnorm(2000), ncol = 4)   # 500 rows
  result_full    <- sigma_heuristic(X, sample_size = 500L, seed = 7L)
  result_sampled <- sigma_heuristic(X, sample_size = 20L,  seed = 7L)
  # Both are positive scalars; the subsampled result differs from the full one
  expect_gt(result_full,    0)
  expect_gt(result_sampled, 0)
  expect_false(isTRUE(all.equal(result_full, result_sampled)))
})

test_that("sigma_heuristic() respects the seed argument", {
  X <- matrix(rnorm(2000), ncol = 4)
  r1 <- sigma_heuristic(X, sample_size = 20L, seed = 99L)
  r2 <- sigma_heuristic(X, sample_size = 20L, seed = 99L)
  r3 <- sigma_heuristic(X, sample_size = 20L, seed = 42L)
  expect_equal(r1, r2)
  expect_false(isTRUE(all.equal(r1, r3)))
})

test_that("rbf_sigma_psvr() returns a dials param with default range [-3, 1]", {
  p <- rbf_sigma_psvr()
  expect_s3_class(p, "quant_param")
  r <- dials::range_get(p, original = FALSE)
  expect_equal(r$lower, -3)
  expect_equal(r$upper, 1)
})

test_that("rbf_sigma_psvr() has a finalize function", {
  p <- rbf_sigma_psvr()
  expect_true(is.function(p$finalize))
})

test_that("cost_psvr() returns a dials param with range [-2, 10]", {
  p <- cost_psvr()
  expect_s3_class(p, "quant_param")
  r <- dials::range_get(p, original = FALSE)
  expect_equal(r$lower, -2)
  expect_equal(r$upper, 10)
})
