## R-vs-C++ kernel parity.
##
## Two tiers, see helper-fp-tiers.R. The strict tier asserts bit-equality
## and runs under skip_on_cran(); the tolerance tier asserts agreement to
## PSVR_FP_TOL and runs everywhere. The polynomial kernel is the reason the
## split exists: R's `^` uses repeated multiplication for integer exponents
## while kernel_poly_cpp uses `powl`, so the two differ by 1-2 ULP on every
## platform tested.
##
## Matrices are built through psvr_memo() so both tiers assert over the
## same values without recomputing the (slow) legacy R nested loop.

## ---- Memoised parity cases -------------------------------------------------

.parity_rbf <- function(N) psvr_memo(sprintf("rbf-%d", N), {
  K <- make_kernel("rbf", sigma = 1.3)
  set.seed(42 + N)
  X <- matrix(rnorm(N * 5), N, 5)
  list(leg = psvr:::.legacy_kernel_matrix(K, X, X),
       cpp = psvr:::kernel_rbf_cpp(X, X, 1.3))
})

.parity_lin <- function(N) psvr_memo(sprintf("lin-%d", N), {
  K <- make_kernel("linear")
  set.seed(42 + N)
  X <- matrix(rnorm(N * 5), N, 5)
  list(leg = psvr:::.legacy_kernel_matrix(K, X, X),
       cpp = psvr:::kernel_linear_cpp(X, X))
})

.parity_poly <- function(N, deg) psvr_memo(sprintf("poly-%d-%d", N, deg), {
  K <- make_kernel("polynomial", degree = deg, coef0 = 0.7)
  set.seed(42 + N + deg)
  X <- matrix(rnorm(N * 5), N, 5)
  list(leg = psvr:::.legacy_kernel_matrix(K, X, X),
       cpp = psvr:::kernel_poly_cpp(X, X, 0.7, deg))
})

.parity_asym <- function() psvr_memo("asym", {
  set.seed(11)
  X1 <- matrix(rnorm(20 * 5), 20, 5)
  X2 <- matrix(rnorm(35 * 5), 35, 5)
  K_rbf  <- make_kernel("rbf", sigma = 1)
  K_lin  <- make_kernel("linear")
  K_poly <- make_kernel("polynomial", degree = 3, coef0 = 1)
  list(
    rbf  = list(cpp = psvr:::kernel_rbf_cpp(X1, X2, 1),
                leg = psvr:::.legacy_kernel_matrix(K_rbf, X1, X2)),
    lin  = list(cpp = psvr:::kernel_linear_cpp(X1, X2),
                leg = psvr:::.legacy_kernel_matrix(K_lin, X1, X2)),
    poly = list(cpp = psvr:::kernel_poly_cpp(X1, X2, 1, 3L),
                leg = psvr:::.legacy_kernel_matrix(K_poly, X1, X2))
  )
})

.parity_crossneg <- function() psvr_memo("crossneg", {
  set.seed(13)
  X <- matrix(rnorm(50 * 4), 50, 4)
  K <- make_kernel("rbf", sigma = 1)
  list(cpp = psvr:::kernel_rbf_cpp(X, -X, 1),
       leg = psvr:::.legacy_kernel_matrix(K, X, -X))
})

.parity_dispatch <- function() psvr_memo("dispatch", {
  set.seed(17)
  X <- matrix(rnorm(50 * 4), 50, 4)
  K_rbf  <- make_kernel("rbf", sigma = 1.5)
  K_lin  <- make_kernel("linear")
  K_poly <- make_kernel("polynomial", degree = 3, coef0 = 0.5)
  list(
    rbf  = list(disp = psvr:::kernel_matrix(K_rbf, X),
                direct = psvr:::kernel_rbf_cpp(X, X, 1.5),
                leg = psvr:::.legacy_kernel_matrix(K_rbf, X, X)),
    lin  = list(disp = psvr:::kernel_matrix(K_lin, X),
                direct = psvr:::kernel_linear_cpp(X, X),
                leg = psvr:::.legacy_kernel_matrix(K_lin, X, X)),
    poly = list(disp = psvr:::kernel_matrix(K_poly, X),
                direct = psvr:::kernel_poly_cpp(X, X, 0.5, 3L),
                leg = psvr:::.legacy_kernel_matrix(K_poly, X, X))
  )
})

.parity_predshape <- function() psvr_memo("predshape", {
  set.seed(23)
  X     <- matrix(rnorm(40 * 4), 40, 4)
  x_new <- matrix(rnorm(4), nrow = 1, ncol = 4)
  K     <- make_kernel("rbf", sigma = 1)
  list(disp = psvr:::kernel_matrix(K, X, x_new),
       leg  = psvr:::.legacy_kernel_matrix(K, X, x_new))
})

.parity_symmat <- function(a) psvr_memo(sprintf("symmat-%d", a), {
  set.seed(29)
  X <- matrix(rnorm(50 * 4), 50, 4)
  K <- make_kernel("rbf", sigma = 1)
  Omega    <- psvr:::.legacy_kernel_matrix(K, X,  X)
  OmegaNeg <- psvr:::.legacy_kernel_matrix(K, X, -X)
  list(disp = psvr:::sym_kernel_matrix(K, X, a),
       ref  = 0.5 * (Omega + a * OmegaNeg))
})

## ---- Strict tier (bit-identical; intra-platform regression gate) ----------

test_that("Rcpp kernels are bit-identical to legacy R nested loop (RBF)", {
  skip_on_cran()
  for (N in c(10L, 100L, 500L)) {
    p <- .parity_rbf(N)
    expect_identical(p$leg, p$cpp, label = sprintf("rbf N=%d", N))
  }
})

test_that("Rcpp kernels are bit-identical to legacy R nested loop (linear)", {
  skip_on_cran()
  for (N in c(10L, 100L, 500L)) {
    p <- .parity_lin(N)
    expect_identical(p$leg, p$cpp, label = sprintf("linear N=%d", N))
  }
})

test_that("Rcpp kernels are bit-identical to legacy R nested loop (polynomial)", {
  skip_on_cran()
  for (deg in c(1L, 2L, 3L, 4L)) {
    for (N in c(10L, 100L, 200L)) {
      p <- .parity_poly(N, deg)
      expect_identical(p$leg, p$cpp,
                       label = sprintf("poly deg=%d N=%d", deg, N))
    }
  }
})

test_that("Rcpp kernels are bit-identical to legacy on asymmetric X1 vs X2", {
  skip_on_cran()
  p <- .parity_asym()
  expect_identical(p$rbf$cpp,  p$rbf$leg)
  expect_identical(p$lin$cpp,  p$lin$leg)
  expect_identical(p$poly$cpp, p$poly$leg)
})

test_that("Rcpp kernels are bit-identical on the cross-negation K(X, -X)", {
  skip_on_cran()
  p <- .parity_crossneg()
  expect_identical(p$cpp, p$leg)
})

test_that("kernel_matrix() dispatch is bit-identical to the legacy reference", {
  skip_on_cran()
  p <- .parity_dispatch()
  for (k in names(p)) {
    expect_identical(p[[k]]$disp, p[[k]]$leg, label = sprintf("%s dispatch", k))
  }
})

test_that("kernel_matrix() predict-shape is bit-identical to legacy", {
  skip_on_cran()
  p <- .parity_predshape()
  expect_identical(p$disp, p$leg)
})

test_that("sym_kernel_matrix() through dispatch is bit-identical for a in {-1, 1}", {
  skip_on_cran()
  for (a in c(-1L, 1L)) {
    p <- .parity_symmat(a)
    expect_identical(p$disp, p$ref, label = sprintf("a=%d", a))
  }
})

## ---- Tolerance tier (runs on every platform, including CRAN) --------------

test_that("Rcpp kernels agree with legacy R to PSVR_FP_TOL (RBF)", {
  for (N in c(10L, 100L, 500L)) {
    p <- .parity_rbf(N)
    expect_equal(p$leg, p$cpp, tolerance = PSVR_FP_TOL,
                 label = sprintf("rbf N=%d", N))
  }
})

test_that("Rcpp kernels agree with legacy R to PSVR_FP_TOL (linear)", {
  for (N in c(10L, 100L, 500L)) {
    p <- .parity_lin(N)
    expect_equal(p$leg, p$cpp, tolerance = PSVR_FP_TOL,
                 label = sprintf("linear N=%d", N))
  }
})

test_that("Rcpp kernels agree with legacy R to PSVR_FP_TOL (polynomial)", {
  # The 1-2 ULP `powl` vs repeated-multiplication difference lives here.
  for (deg in c(1L, 2L, 3L, 4L)) {
    for (N in c(10L, 100L, 200L)) {
      p <- .parity_poly(N, deg)
      expect_equal(p$leg, p$cpp, tolerance = PSVR_FP_TOL,
                   label = sprintf("poly deg=%d N=%d", deg, N))
    }
  }
})

test_that("Rcpp kernels agree with legacy to PSVR_FP_TOL on asymmetric X1 vs X2", {
  p <- .parity_asym()
  expect_equal(p$rbf$cpp,  p$rbf$leg,  tolerance = PSVR_FP_TOL)
  expect_equal(p$lin$cpp,  p$lin$leg,  tolerance = PSVR_FP_TOL)
  expect_equal(p$poly$cpp, p$poly$leg, tolerance = PSVR_FP_TOL)
})

test_that("Rcpp kernels agree to PSVR_FP_TOL on the cross-negation K(X, -X)", {
  p <- .parity_crossneg()
  expect_equal(p$cpp, p$leg, tolerance = PSVR_FP_TOL)
})

test_that("kernel_matrix() dispatch agrees with legacy to PSVR_FP_TOL", {
  p <- .parity_dispatch()
  for (k in names(p)) {
    expect_equal(p[[k]]$disp, p[[k]]$leg, tolerance = PSVR_FP_TOL,
                 label = sprintf("%s dispatch", k))
  }
})

test_that("kernel_matrix() predict-shape agrees with legacy to PSVR_FP_TOL", {
  p <- .parity_predshape()
  expect_equal(p$disp, p$leg, tolerance = PSVR_FP_TOL)
})

test_that("sym_kernel_matrix() dispatch agrees to PSVR_FP_TOL for a in {-1, 1}", {
  for (a in c(-1L, 1L)) {
    p <- .parity_symmat(a)
    expect_equal(p$disp, p$ref, tolerance = PSVR_FP_TOL,
                 label = sprintf("a=%d", a))
  }
})

## ---- Platform-independent assertions (no tiering needed) ------------------
## Rcpp-vs-Rcpp and R-vs-R comparisons run the same code on both sides, so
## they are bit-equal on every platform by construction.

test_that("kernel_matrix() dispatched path == direct Rcpp call", {
  p <- .parity_dispatch()
  expect_identical(p$rbf$disp,  p$rbf$direct)
  expect_identical(p$lin$disp,  p$lin$direct)
  expect_identical(p$poly$disp, p$poly$direct)
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
  p <- .parity_asym()
  expect_equal(dim(p$rbf$cpp),  c(20L, 35L))
  expect_equal(dim(p$lin$cpp),  c(20L, 35L))
  expect_equal(dim(p$poly$cpp), c(20L, 35L))
})

test_that("kernel_matrix() handles predict-shape (single test row)", {
  p <- .parity_predshape()
  expect_equal(dim(p$disp), c(40L, 1L))
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
