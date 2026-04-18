#' Fit epsilon-SVR with MAPE loss (Model 1)
#'
#' Solves the quadratic program derived in Theorem 1 of Benavides-Herrera et al.
#' (2026) via `osqp`. The dual is expressed in variables
#' `u = [α; α*] ∈ R^{2N}` with:
#'
#' - **P** = `[Ω, -Ω; -Ω, Ω]` (2N × 2N, upper triangular passed to osqp)
#' - **q** = `[y(ε/100 − 1); y(1 + ε/100)]`
#' - **Equality:** `[1ᵀ, −1ᵀ] u = 0`
#' - **Box:** `0 ≤ αk ≤ 100C/yk`, `0 ≤ αk* ≤ 100C/yk`
#'
#' After solving, `βk = αk − αk*` and only support vectors with `|βk| > tol`
#' are retained.
#'
#' @param X Numeric matrix of training inputs, one observation per row (N × p).
#' @param y Numeric vector of training targets, length N. Must satisfy `y > 0`.
#' @param kernel A kernel function created by [make_kernel()].
#' @param C Regularization parameter `C > 0`.
#' @param eps Insensitivity tube half-width `ε ≥ 0` (in percentage units,
#'   i.e., the tube is `ε%` of each target).
#' @param tol Threshold below which `|βk|` is treated as zero when selecting
#'   support vectors and free support vectors (default `1e-5`).
#'
#' @return An object of class `"psvr_mape"`, a list with components:
#'   \describe{
#'     \item{`beta`}{Numeric vector of non-zero dual differences `βk` (length
#'       equal to the number of support vectors).}
#'     \item{`b`}{Numeric scalar bias term.}
#'     \item{`X_sv`}{Numeric matrix of support vector inputs.}
#'     \item{`kernel`}{The kernel function used.}
#'     \item{`eps`}{The `ε` value used.}
#'   }
#'
#' @examples
#' X <- matrix(1:6, ncol = 2)
#' y <- c(2.1, 3.8, 6.2)
#' K <- make_kernel("rbf", sigma = 1)
#' fit <- mape_svr(X, y, kernel = K, C = 10, eps = 5)
#' predict(fit, X)
#'
#' @export
mape_svr <- function(X, y, kernel, C, eps, tol = 1e-5) {
  X <- as.matrix(X)
  y <- as.numeric(y)
  if (!all(y > 0)) stop("all targets `y` must be strictly positive")
  if (C   <= 0)    stop("`C` must be positive")
  if (eps <  0)    stop("`eps` must be non-negative")

  N     <- nrow(X)
  scale <- eps / 100          # ε/100, used throughout
  ub    <- 100 * C / y        # per-sample upper bounds on αk and αk*

  Omega <- kernel_matrix(kernel, X)

  # ---- QP matrices ----
  # P = [Ω, -Ω; -Ω, Ω], upper triangular for osqp
  P_dense <- rbind(cbind(Omega, -Omega), cbind(-Omega, Omega))
  P       <- Matrix::triu(Matrix::Matrix(P_dense, sparse = TRUE))

  # q = [y(ε/100 - 1); y(1 + ε/100)]
  q <- c(y * (scale - 1.0), y * (1.0 + scale))

  # ---- Constraint matrix ----
  # Row 1:        [1ᵀ, -1ᵀ] u = 0   (equality: Σαk = Σαk*)
  # Rows 2..2N+1: I_{2N} u ∈ [0, ub] (per-variable box)
  A_eq  <- Matrix::Matrix(matrix(c(rep(1.0, N), rep(-1.0, N)), nrow = 1L),
                           sparse = TRUE)
  A_box <- Matrix::Diagonal(2L * N)
  A     <- rbind(A_eq, A_box)

  l <- c(0.0, rep(0.0, 2L * N))
  u <- c(0.0, rep(ub,  2L))        # ub recycled: [ub for α; ub for α*]

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
  # Ω·β = N-vector of f(xk) - b values
  Kbeta <- as.numeric(Omega %*% beta)

  free_up <- which(alpha      > tol & alpha      < ub - tol)
  free_lo <- which(alpha_star > tol & alpha_star < ub - tol)

  b_up <- y[free_up] * (1.0 - scale) - Kbeta[free_up]  # from yk - f(xk) = ε*yk/100
  b_lo <- y[free_lo] * (1.0 + scale) - Kbeta[free_lo]  # from f(xk) - yk = ε*yk/100

  b_vals <- c(b_up, b_lo)

  if (length(b_vals) == 0L) {
    # No free SVs: sandwich b between KKT bounds from saturated SVs
    sat_up <- which(alpha      > tol)   # αk = ub  → b ≤ yk*(1-ε/100) - Kbeta[k]
    sat_lo <- which(alpha_star > tol)   # αk*= ub  → b ≥ yk*(1+ε/100) - Kbeta[k]
    bub_bound <- if (length(sat_up) > 0L) min(y[sat_up] * (1.0 - scale) - Kbeta[sat_up]) else  Inf
    blb_bound <- if (length(sat_lo) > 0L) max(y[sat_lo] * (1.0 + scale) - Kbeta[sat_lo]) else -Inf
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
      eps    = eps
    ),
    class = "psvr_mape"
  )
}

#' Predict from a fitted epsilon-SVR with MAPE model
#'
#' @param object An object of class `"psvr_mape"` from [mape_svr()].
#' @param newdata Numeric matrix of new inputs, one observation per row (M × p).
#' @param ... Ignored.
#'
#' @return Numeric vector of length M with predicted values.
#'
#' @examples
#' X <- matrix(1:6, ncol = 2)
#' y <- c(2.1, 3.8, 6.2)
#' K <- make_kernel("rbf", sigma = 1)
#' fit <- mape_svr(X, y, kernel = K, C = 10, eps = 5)
#' predict(fit, X)
#'
#' @export
predict.psvr_mape <- function(object, newdata, ...) {
  newdata <- as.matrix(newdata)
  M       <- nrow(newdata)
  preds   <- numeric(M)
  for (i in seq_len(M)) {
    kv        <- kernel_matrix(object$kernel, object$X_sv,
                               newdata[i, , drop = FALSE])
    preds[i]  <- sum(object$beta * kv) + object$b
  }
  preds
}
