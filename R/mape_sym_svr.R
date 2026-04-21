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
#'     \item{`y_sv`}{Numeric vector of support vector targets.}
#'     \item{`kernel`}{The kernel function used.}
#'     \item{`C`}{The regularization parameter `C`.}
#'     \item{`eps`}{The `ε` value used.}
#'     \item{`a`}{The symmetry parameter.}
#'     \item{`n_train`}{Number of training observations.}
#'     \item{`p_train`}{Number of training features (columns).}
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
  if (!all(y > 0)) {
    n_bad <- sum(y <= 0)
    stop(sprintf(
      paste0("%d target value%s non-positive (min = %g). ",
             "All targets must be strictly positive for percentage-error loss."),
      n_bad, if (n_bad == 1L) " is" else "s are", min(y)
    ))
  }
  if (C   <= 0)          stop("`C` must be positive")
  if (eps <  0)          stop("`eps` must be non-negative")
  if (!a %in% c(-1L, 1L)) stop("`a` must be 1 (even) or -1 (odd)")

  N     <- nrow(X)
  if (N > 2000L) {
    warning(sprintf(
      paste0("Large dataset (N = %d): kernel matrix is %d x %d (%.1f MB). ",
             "Consider subsampling for hyperparameter tuning."),
      N, N, N, N^2 * 8 / 1e6
    ))
  }
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
  res <- osqp::solve_osqp(P, q, A, l, u, pars = settings)

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
      beta    = beta[sv_idx],
      b       = b,
      X_sv    = X[sv_idx, , drop = FALSE],
      y_sv    = y[sv_idx],
      kernel  = kernel,
      C       = C,
      eps     = eps,
      a       = a,
      n_train = N,
      p_train = ncol(X)
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
  p <- ncol(newdata)
  if (p != object$p_train)
    stop(sprintf("newdata has %d column%s but model was trained on %d",
                 p, if (p == 1L) "" else "s", object$p_train))
  M     <- nrow(newdata)
  preds <- numeric(M)
  for (i in seq_len(M)) {
    # sym_kernel_vector returns ½·Ks(xk, x) for each support vector k
    kv       <- sym_kernel_vector(object$kernel, object$X_sv,
                                  newdata[i, ], object$a)
    preds[i] <- sum(object$beta * kv) + object$b
  }
  as.numeric(preds)
}

#' Print method for psvr_mape_sym objects
#'
#' @param x An object of class `"psvr_mape_sym"`.
#' @param ... Ignored.
#'
#' @return `x`, invisibly.
#'
#' @export
print.psvr_mape_sym <- function(x, ...) {
  ki    <- attr(x$kernel, "kernel_info")
  kdesc <- .kernel_desc(ki)
  nsv   <- length(x$beta)
  sym   <- if (x$a == 1L) "even  (a = 1)" else "odd   (a = -1)"
  cat(sprintf(
    "\nSymmetric epsilon-SVR with MAPE loss  [psvr_mape_sym]\n\n  Kernel:          %s\n  C:               %g\n  eps:             %g\n  Symmetry:        %s\n  Training obs.:   %d\n  Support vectors: %d (%.1f%%)\n\n",
    kdesc, x$C, x$eps, sym, x$n_train, nsv, 100 * nsv / x$n_train
  ))
  invisible(x)
}

#' Extract coefficients from a psvr_mape_sym model
#'
#' @param object An object of class `"psvr_mape_sym"`.
#' @param ... Ignored.
#'
#' @return A named list with components:
#'   \describe{
#'     \item{`alpha`}{Dual variable differences `βk = αk − αk*` for support vectors.}
#'     \item{`b`}{Bias term.}
#'     \item{`X_sv`}{Support vector input matrix.}
#'   }
#'
#' @export
coef.psvr_mape_sym <- function(object, ...) {
  list(alpha = object$beta, b = object$b, X_sv = object$X_sv)
}
