#' Create a kernel function
#'
#' Returns a closure `K(xi, xj)` suitable for use with all four psvr model
#' fitting functions. The returned function accepts numeric vectors of any sign,
#' which is required by the symmetric models (Models 2 and 4) that evaluate
#' `K(xk, -xl)`.
#'
#' @param type Kernel type: `"rbf"`, `"linear"`, or `"polynomial"`.
#' @param sigma Bandwidth for the RBF kernel, `sigma > 0` (default `1`).
#' @param degree Integer degree for the polynomial kernel, `degree >= 1`
#'   (default `3`).
#' @param coef0 Constant term for the polynomial kernel (default `1`).
#'
#' @return A function `K(xi, xj)` where `xi` and `xj` are numeric vectors of
#'   the same length, returning a scalar kernel evaluation.
#'
#' @details
#' The three supported kernels are:
#' - **RBF:** `K(xi, xj) = exp(-‖xi - xj‖² / (2 * sigma²))`
#' - **Linear:** `K(xi, xj) = xi · xj`
#' - **Polynomial:** `K(xi, xj) = (xi · xj + coef0)^degree`
#'
#' RBF and even-degree polynomial kernels satisfy Assumption 3 of the paper
#' (kernel symmetry), making them compatible with the symmetric models.
#' The linear kernel and odd-degree polynomial kernels do **not** satisfy
#' Assumption 3 and should not be used with `mape_sym_svr()` or
#' `rmspe_sym_lssvr()`.
#'
#' @examples
#' K <- make_kernel("rbf", sigma = 0.5)
#' K(c(1, 2), c(3, 4))
#'
#' K_lin <- make_kernel("linear")
#' K_lin(c(1, 0), c(0, 1))
#'
#' K_poly <- make_kernel("polynomial", degree = 2, coef0 = 1)
#' K_poly(c(1, 2), c(3, 4))
#'
#' @export
make_kernel <- function(type = c("rbf", "linear", "polynomial"),
                        sigma = 1, degree = 3L, coef0 = 1) {
  type <- match.arg(type)
  switch(
    type,
    rbf = {
      if (sigma <= 0) stop("`sigma` must be positive")
      sigma2 <- 2 * sigma^2
      function(xi, xj) {
        d <- xi - xj
        exp(-sum(d * d) / sigma2)
      }
    },
    linear = {
      function(xi, xj) sum(xi * xj)
    },
    polynomial = {
      degree <- as.integer(degree)
      if (degree < 1L) stop("`degree` must be >= 1")
      function(xi, xj) (sum(xi * xj) + coef0)^degree
    }
  )
}

#' Compute a kernel matrix between two sets of points
#'
#' Entry `[i, j]` equals `K(X1[i, ], X2[j, ])`. Used internally by all four
#' model fitting and prediction functions.
#'
#' @param K A kernel function from [make_kernel()].
#' @param X1 Numeric matrix with one observation per row (n1 × p).
#' @param X2 Numeric matrix with one observation per row (n2 × p).
#'   Defaults to `X1`, giving the square training kernel matrix Ω.
#'
#' @return Numeric matrix of size n1 × n2.
#'
#' @keywords internal
kernel_matrix <- function(K, X1, X2 = X1) {
  n1 <- nrow(X1)
  n2 <- nrow(X2)
  M <- matrix(0.0, n1, n2)
  for (i in seq_len(n1)) {
    for (j in seq_len(n2)) {
      M[i, j] <- K(X1[i, ], X2[j, ])
    }
  }
  M
}

#' Compute the symmetrized kernel matrix Ωs = ½(Ω + a·Ω*)
#'
#' Used by the symmetric LS-SVR model (Model 4). Entry `[k, l]` of `Ω*` is
#' `K(xk, -xl)`, so negation is applied to the columns of X (i.e., to `X2`).
#'
#' @param K A kernel function from [make_kernel()].
#' @param X Numeric training matrix (N × p).
#' @param a Symmetry parameter: `1` (even) or `-1` (odd).
#'
#' @return Numeric N × N matrix `Ωs = ½(Ω + a·Ω*)`.
#'
#' @keywords internal
sym_kernel_matrix <- function(K, X, a) {
  Omega     <- kernel_matrix(K, X, X)
  Omega_neg <- kernel_matrix(K, X, -X)
  0.5 * (Omega + a * Omega_neg)
}

#' Compute a symmetric kernel vector for prediction
#'
#' For a new point `x`, returns the N-vector with entry `k` equal to
#' `½ * Ks(X[k, ], x)` where `Ks(xi, xj) = K(xi, xj) + a * K(xi, -xj)`.
#' Used by `predict.psvr_mape_sym()` and `predict.psvr_rmspe_sym()`.
#'
#' @param K A kernel function from [make_kernel()].
#' @param X Numeric training matrix (N × p).
#' @param x Numeric vector (length p), the new point to predict.
#' @param a Symmetry parameter: `1` (even) or `-1` (odd).
#'
#' @return Numeric vector of length N.
#'
#' @keywords internal
sym_kernel_vector <- function(K, X, x, a) {
  n <- nrow(X)
  v <- numeric(n)
  for (k in seq_len(n)) {
    v[k] <- 0.5 * (K(X[k, ], x) + a * K(X[k, ], -x))
  }
  v
}
