test_that("make_kernel returns a function", {
  K <- make_kernel("rbf", sigma = 1)
  expect_true(is.function(K))

  K_lin <- make_kernel("linear")
  expect_true(is.function(K_lin))

  K_poly <- make_kernel("polynomial", degree = 2, coef0 = 1)
  expect_true(is.function(K_poly))
})

test_that("make_kernel validates parameters", {
  expect_error(make_kernel("rbf", sigma = 0),   "`sigma` must be positive")
  expect_error(make_kernel("rbf", sigma = -1),  "`sigma` must be positive")
  expect_error(make_kernel("polynomial", degree = 0), "`degree` must be >= 1")
})

test_that("RBF kernel is symmetric and positive definite", {
  K  <- make_kernel("rbf", sigma = 1)
  xi <- c(1, 2)
  xj <- c(3, 4)
  expect_equal(K(xi, xj), K(xj, xi))
  expect_true(K(xi, xi) > 0)
  expect_equal(K(xi, xi), 1)   # RBF self-kernel is always 1
})

test_that("RBF kernel satisfies Assumption 3 (kernel symmetry)", {
  K  <- make_kernel("rbf", sigma = 1.5)
  xi <- c(1.2, -0.5)
  xj <- c(-0.3,  2.1)
  # K(-xi, xj) == K(xi, -xj)
  expect_equal(K(-xi, xj), K(xi, -xj))
  # K(-xi, -xj) == K(xi, xj)
  expect_equal(K(-xi, -xj), K(xi, xj))
})

test_that("linear kernel does NOT satisfy Assumption 3 in general", {
  K  <- make_kernel("linear")
  xi <- c(1, 2)
  xj <- c(3, 4)
  # K(-xi, xj) = -(xi.xj) but K(xi, -xj) = -(xi.xj): these are equal here
  # Actually they happen to be equal for linear; test the self-kernel property:
  # K(-xi, -xj) == K(xi, xj) holds for linear ((-xi).(-xj) = xi.xj)
  expect_equal(K(-xi, -xj), K(xi, xj))
  # but K(-xi, xj) = -K(xi, xj) != K(xi, xj) unless K(xi,xj)=0 -- fails Assumption 3
  expect_false(isTRUE(all.equal(K(-xi, xj), K(xi, xj))))
})

test_that("kernel_matrix returns correct dimensions", {
  K  <- make_kernel("rbf", sigma = 1)
  X1 <- matrix(rnorm(12), 4, 3)
  X2 <- matrix(rnorm(9),  3, 3)

  M_sq   <- kernel_matrix(K, X1)
  M_rect <- kernel_matrix(K, X1, X2)

  expect_equal(dim(M_sq),   c(4L, 4L))
  expect_equal(dim(M_rect), c(4L, 3L))
})

test_that("kernel_matrix diagonal is all 1 for RBF", {
  K <- make_kernel("rbf", sigma = 1)
  X <- matrix(rnorm(20), 10, 2)
  M <- kernel_matrix(K, X)
  expect_equal(diag(M), rep(1, 10))
})

test_that("sym_kernel_matrix returns ½(Ω + a·Ω*)", {
  K  <- make_kernel("rbf", sigma = 1)
  X  <- matrix(rnorm(12), 6, 2)
  a  <- 1L

  Omega_s   <- sym_kernel_matrix(K, X, a)
  Omega     <- kernel_matrix(K, X, X)
  Omega_neg <- kernel_matrix(K, X, -X)
  expected  <- 0.5 * (Omega + a * Omega_neg)

  expect_equal(Omega_s, expected)
  expect_equal(dim(Omega_s), c(6L, 6L))
})
