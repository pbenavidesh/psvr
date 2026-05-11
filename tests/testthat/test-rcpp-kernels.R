test_that("Rcpp kernels are bit-identical to legacy R nested loop (RBF)", {
  K_rbf <- make_kernel("rbf", sigma = 1.3)
  for (N in c(10L, 100L, 500L)) {
    set.seed(42 + N)
    X <- matrix(rnorm(N * 5), N, 5)
    M_leg <- psvr:::.legacy_kernel_matrix(K_rbf, X, X)
    M_cpp <- psvr:::kernel_rbf_cpp(X, X, 1.3)
    expect_identical(M_leg, M_cpp)
  }
})

test_that("Rcpp kernels are bit-identical to legacy R nested loop (linear)", {
  K_lin <- make_kernel("linear")
  for (N in c(10L, 100L, 500L)) {
    set.seed(42 + N)
    X <- matrix(rnorm(N * 5), N, 5)
    M_leg <- psvr:::.legacy_kernel_matrix(K_lin, X, X)
    M_cpp <- psvr:::kernel_linear_cpp(X, X)
    expect_identical(M_leg, M_cpp)
  }
})

test_that("Rcpp kernels are bit-identical to legacy R nested loop (polynomial)", {
  for (deg in c(1L, 2L, 3L, 4L)) {
    K_poly <- make_kernel("polynomial", degree = deg, coef0 = 0.7)
    for (N in c(10L, 100L, 200L)) {
      set.seed(42 + N + deg)
      X <- matrix(rnorm(N * 5), N, 5)
      M_leg <- psvr:::.legacy_kernel_matrix(K_poly, X, X)
      M_cpp <- psvr:::kernel_poly_cpp(X, X, 0.7, deg)
      expect_identical(M_leg, M_cpp)
    }
  }
})

test_that("Rcpp self-kernel is exactly symmetric (RBF)", {
  set.seed(7)
  X <- matrix(rnorm(100 * 5), 100, 5)
  M <- psvr:::kernel_rbf_cpp(X, X, 1.0)
  expect_equal(max(abs(M - t(M))), 0)
})

test_that("Rcpp self-kernel is exactly symmetric (linear, polynomial)", {
  set.seed(7)
  X <- matrix(rnorm(100 * 5), 100, 5)
  M_lin  <- psvr:::kernel_linear_cpp(X, X)
  M_poly <- psvr:::kernel_poly_cpp(X, X, 1.0, 3L)
  expect_equal(max(abs(M_lin  - t(M_lin))),  0)
  expect_equal(max(abs(M_poly - t(M_poly))), 0)
})

test_that("Rcpp kernels handle asymmetric X1 vs X2 with correct dim", {
  set.seed(11)
  X1 <- matrix(rnorm(20 * 5), 20, 5)
  X2 <- matrix(rnorm(35 * 5), 35, 5)
  K_rbf  <- make_kernel("rbf", sigma = 1)
  K_lin  <- make_kernel("linear")
  K_poly <- make_kernel("polynomial", degree = 3, coef0 = 1)

  M_rbf  <- psvr:::kernel_rbf_cpp(X1, X2, 1)
  M_lin  <- psvr:::kernel_linear_cpp(X1, X2)
  M_poly <- psvr:::kernel_poly_cpp(X1, X2, 1, 3L)

  expect_equal(dim(M_rbf),  c(20L, 35L))
  expect_equal(dim(M_lin),  c(20L, 35L))
  expect_equal(dim(M_poly), c(20L, 35L))

  # Bit-identical to legacy
  expect_identical(M_rbf,  psvr:::.legacy_kernel_matrix(K_rbf,  X1, X2))
  expect_identical(M_lin,  psvr:::.legacy_kernel_matrix(K_lin,  X1, X2))
  expect_identical(M_poly, psvr:::.legacy_kernel_matrix(K_poly, X1, X2))
})

test_that("Rcpp kernels handle the cross-negation K(X, -X) used by sym models", {
  set.seed(13)
  X <- matrix(rnorm(50 * 4), 50, 4)
  K <- make_kernel("rbf", sigma = 1)
  expect_identical(psvr:::kernel_rbf_cpp(X, -X, 1),
                   psvr:::.legacy_kernel_matrix(K, X, -X))
})

test_that("Rcpp kernels error on mismatched ncol", {
  X1 <- matrix(rnorm(20), 5, 4)
  X2 <- matrix(rnorm(15), 5, 3)
  expect_error(psvr:::kernel_rbf_cpp(X1, X2, 1),
               "same number of columns")
  expect_error(psvr:::kernel_linear_cpp(X1, X2),
               "same number of columns")
  expect_error(psvr:::kernel_poly_cpp(X1, X2, 1, 3L),
               "same number of columns")
})

test_that("kernel_rbf_cpp errors on non-positive sigma", {
  X <- matrix(rnorm(20), 5, 4)
  expect_error(psvr:::kernel_rbf_cpp(X, X, 0),  "sigma.*positive")
  expect_error(psvr:::kernel_rbf_cpp(X, X, -1), "sigma.*positive")
})

test_that("kernel_poly_cpp errors on degree < 1", {
  X <- matrix(rnorm(20), 5, 4)
  expect_error(psvr:::kernel_poly_cpp(X, X, 1, 0L),  "degree.*>= 1")
  expect_error(psvr:::kernel_poly_cpp(X, X, 1, -2L), "degree.*>= 1")
})

test_that("kernel_matrix() dispatches to Rcpp for make_kernel() closures", {
  set.seed(17)
  X <- matrix(rnorm(50 * 4), 50, 4)

  K_rbf  <- make_kernel("rbf", sigma = 1.5)
  K_lin  <- make_kernel("linear")
  K_poly <- make_kernel("polynomial", degree = 3, coef0 = 0.5)

  # Dispatched path == direct Rcpp
  expect_identical(psvr:::kernel_matrix(K_rbf,  X), psvr:::kernel_rbf_cpp(X, X, 1.5))
  expect_identical(psvr:::kernel_matrix(K_lin,  X), psvr:::kernel_linear_cpp(X, X))
  expect_identical(psvr:::kernel_matrix(K_poly, X), psvr:::kernel_poly_cpp(X, X, 0.5, 3L))

  # Dispatched path == legacy R reference
  expect_identical(psvr:::kernel_matrix(K_rbf,  X), psvr:::.legacy_kernel_matrix(K_rbf,  X, X))
  expect_identical(psvr:::kernel_matrix(K_lin,  X), psvr:::.legacy_kernel_matrix(K_lin,  X, X))
  expect_identical(psvr:::kernel_matrix(K_poly, X), psvr:::.legacy_kernel_matrix(K_poly, X, X))
})

test_that("kernel_matrix() falls through to legacy for user-defined closures", {
  set.seed(19)
  X <- matrix(rnorm(20 * 3), 20, 3)
  K_custom <- function(xi, xj) (sum(xi * xj) + 2)^2
  # No kernel_info attribute -> dispatch falls through to legacy
  expect_null(attr(K_custom, "kernel_info"))
  expect_identical(
    psvr:::kernel_matrix(K_custom, X, X),
    psvr:::.legacy_kernel_matrix(K_custom, X, X)
  )
})

test_that("kernel_matrix() handles predict-shape (single test row)", {
  set.seed(23)
  X     <- matrix(rnorm(40 * 4), 40, 4)
  x_new <- matrix(rnorm(4), nrow = 1, ncol = 4)
  K     <- make_kernel("rbf", sigma = 1)
  kv_disp <- psvr:::kernel_matrix(K, X, x_new)
  kv_leg  <- psvr:::.legacy_kernel_matrix(K, X, x_new)
  expect_identical(kv_disp, kv_leg)
  expect_equal(dim(kv_disp), c(40L, 1L))
})

test_that("sym_kernel_matrix() through dispatch is bit-identical for a in {-1, 1}", {
  set.seed(29)
  X <- matrix(rnorm(50 * 4), 50, 4)
  K <- make_kernel("rbf", sigma = 1)

  for (a in c(-1L, 1L)) {
    M_disp   <- psvr:::sym_kernel_matrix(K, X, a)
    Omega    <- psvr:::.legacy_kernel_matrix(K, X,  X)
    OmegaNeg <- psvr:::.legacy_kernel_matrix(K, X, -X)
    M_ref    <- 0.5 * (Omega + a * OmegaNeg)
    expect_identical(M_disp, M_ref)
  }
})
