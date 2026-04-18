#' Fit symmetric epsilon-SVR with MAPE loss (Model 2)
#'
#' Solves the quadratic program derived in Theorem 2 of Benavides-Herrera et al.
#' (2026) via `osqp`. The symmetry constraint `f(x) = a·f(-x)` is enforced by
#' replacing the kernel with `Ks(xi, xj) = K(xi, xj) + a·K(xi, -xj)`.
#'
#' The dual in variables `u = [α; α*] ∈ R^{2N}` is:
#'
#' - **P** = `½ · [Ks, -Ks; -Ks, Ks]` so that osqp's `½ uᵀPu` evaluates to
#'   `¼ βᵀKsβ`, matching the `−¼` coefficient in Theorem 2.
#' - **q** = `[y(ε/100 − 1); y(1 + ε/100)]` (identical to Model 1)
#' - **Equality:** `[1ᵀ, −1ᵀ] u = 0`
#' - **Box:** `0 ≤ αk ≤ 100C/yk`, `0 ≤ αk* ≤ 100C/yk` (identical to Model 1)
#'
#' Note: `Ks = Ω + a·Ω*` carries **no** `½` factor here. The `½` lives in P
#' so that osqp's internal `½` produces the required `¼` overall.
#' Contrast with [rmspe_sym_lssvr()] (Model 4), which uses
#' `sym_kernel_matrix()` returning `½(Ω + a·Ω*)` = `Ωs`.
#'
#' The kernel must satisfy Assumption 3 of the paper; see [make_kernel()].
#'
#' @param X Numeric matrix of training inputs, one observation per row (N × p).
#' @param y Numeric vector of training targets, length N. Must satisfy `y > 0`.
#' @param kernel A kernel function created by [make_kernel()].
#' @param C Regularization parameter `C > 0`.
#' @param eps Insensitivity tube half-width `ε ≥ 0` (in percentage units).
#' @param a Symmetry parameter: `1` for even symmetry `f(x) = f(-x)`,
#'   `-1` for odd symmetry `f(x) = -f(-x)`.
#' @param tol Threshold below which `|βk|` is treated as zero (default `1e-5`).
#'
#' @return An object of class `"psvr_mape_sym"`, a list with components:
#'   \describe{
#'     \item{`beta`}{Numeric vector of non-zero dual differences `βk`.}
#'     \item{`b`}{Numeric scalar bias term.}
#'     \item{`X_sv`}{Numeric matrix of support vector inputs.}
#'     \item{`kernel`}{The kernel function used.}
#'     \item{`eps`}{The `ε` value used.}
#'     \item{`a`}{The symmetry parameter.}
#'   }
#'
#' @examples
#' X <- matrix(1:6, ncol = 2)
#' y <- c(2.1, 3.8, 6.2)
#' K <- make_kernel("rbf", sigma = 1)
#' fit <- mape_sym_svr(X, y, kernel = K, C = 10, eps = 5, a = 1)
#' predict(fit, X)
#'
#' @export
mape_sym_svr <- function(X, y, kernel, C, eps, a = 1, tol = 1e-5) {
  X <- as.matrix(X)
  y <- as.numeric(y)
  if (!all(y > 0))       stop("all targets `y` must be strictly positive")
  if (C   <= 0)          stop("`C` must be positive")
  if (eps <  0)          stop("`eps` must be non-negative")
  if (!a %in% c(-1L, 1L)) stop("`a` must be 1 (even) or -1 (odd)")

  N     <- nrow(X)
  scale <- eps / 100
  ub    <- 100 * C / y

  # Ks = Ω + a·Ω*  (no ½ — the ½ is absorbed into P below)
  Omega     <- kernel_matrix(kernel, X, X)
  Omega_neg <- kernel_matrix(kernel, X, -X)
  Ks        <- Omega + a * Omega_neg

  # ---- QP matrices ----
  # P = ½[Ks,-Ks;-Ks,Ks] so that ½uᵀPu = ¼βᵀKsβ  (Theorem 2 coefficient)
  P_dense <- 0.5 * rbind(cbind(Ks, -Ks), cbind(-Ks, Ks))
  P       <- Matrix::triu(Matrix::Matrix(P_dense, sparse = TRUE))

  # q identical to Model 1
  q <- c(y * (scale - 1.0), y * (1.0 + scale))

  # ---- Constraint matrix (identical to Model 1) ----
  A_eq  <- Matrix::Matrix(matrix(c(rep(1.0, N), rep(-1.0, N)), nrow = 1L),
                           sparse = TRUE)
  A_box <- Matrix::Diagonal(2L * N)
  A     <- rbind(A_eq, A_box)

  l <- c(0.0, rep(0.0, 2L * N))
  u <- c(0.0, rep(ub,  2L))

  # ---- Solve ----
  settings <- osqp::osqpSettings(
    verbose  = FALSE,
    eps_abs  = 1e-8,
    eps_rel  = 1e-8,
    max_iter = 10000L
  )
  prob <- osqp::osqp(P, q, A, l, u, pars = settings)
  res  <- prob$solve()

  if (!startsWith(res$info$status, "solved")) {
    warning("osqp status: ", res$info$status)
  }

  alpha      <- res$x[seq_len(N)]
  alpha_star <- res$x[seq_len(N) + N]
  beta       <- alpha - alpha_star

  # ---- Recover bias b ----
  # Symmetric representer: f(xk) = ½(Ks·β)[k] + b
  Ksbeta_half <- 0.5 * as.numeric(Ks %*% beta)

  free_up <- which(alpha      > tol & alpha      < ub - tol)
  free_lo <- which(alpha_star > tol & alpha_star < ub - tol)

  # Free upper: yk - f(xk) = ε·yk/100  →  b = yk(1-ε/100) - ½(Ks·β)[k]
  # Free lower: f(xk) - yk = ε·yk/100  →  b = yk(1+ε/100) - ½(Ks·β)[k]
  b_up <- y[free_up] * (1.0 - scale) - Ksbeta_half[free_up]
  b_lo <- y[free_lo] * (1.0 + scale) - Ksbeta_half[free_lo]

  b_vals <- c(b_up, b_lo)

  if (length(b_vals) == 0L) {
    sat_up    <- which(alpha      > tol)
    sat_lo    <- which(alpha_star > tol)
    bub_bound <- if (length(sat_up) > 0L) min(y[sat_up] * (1.0 - scale) - Ksbeta_half[sat_up]) else  Inf
    blb_bound <- if (length(sat_lo) > 0L) max(y[sat_lo] * (1.0 + scale) - Ksbeta_half[sat_lo]) else -Inf
    b <- if (is.finite(bub_bound) && is.finite(blb_bound)) (bub_bound + blb_bound) / 2
         else if (is.finite(bub_bound)) bub_bound
         else if (is.finite(blb_bound)) blb_bound
         else 0.0
  } else {
    b <- mean(b_vals)
  }

  # ---- Retain support vectors only ----
  sv_idx <- which(abs(beta) > tol)

  structure(
    list(
      beta   = beta[sv_idx],
      b      = b,
      X_sv   = X[sv_idx, , drop = FALSE],
      kernel = kernel,
      eps    = eps,
      a      = a
    ),
    class = "psvr_mape_sym"
  )
}

#' Predict from a fitted symmetric epsilon-SVR with MAPE model
#'
#' Uses the symmetric representer theorem:
#' `f(x) = ½ Σk βk Ks(xk, x) + b`
#' where `Ks(xk, x) = K(xk, x) + a·K(xk, -x)`.
#'
#' @param object An object of class `"psvr_mape_sym"` from [mape_sym_svr()].
#' @param newdata Numeric matrix of new inputs, one observation per row (M × p).
#' @param ... Ignored.
#'
#' @return Numeric vector of length M with predicted values.
#'
#' @examples
#' X <- matrix(1:6, ncol = 2)
#' y <- c(2.1, 3.8, 6.2)
#' K <- make_kernel("rbf", sigma = 1)
#' fit <- mape_sym_svr(X, y, kernel = K, C = 10, eps = 5, a = 1)
#' predict(fit, X)
#'
#' @export
predict.psvr_mape_sym <- function(object, newdata, ...) {
  newdata <- as.matrix(newdata)
  M       <- nrow(newdata)
  preds   <- numeric(M)
  for (i in seq_len(M)) {
    # sym_kernel_vector returns ½·Ks(xk, x) for each support vector k
    kv       <- sym_kernel_vector(object$kernel, object$X_sv,
                                  newdata[i, ], object$a)
    preds[i] <- sum(object$beta * kv) + object$b
  }
  preds
}
