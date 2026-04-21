#' Fit symmetric LS-SVR with RMSPE loss (Model 4)
#'
#' Solves the linear system derived in Theorem 4 of Benavides-Herrera et al.
#' (2026). Identical in structure to [rmspe_lssvr()] (Model 3) but replaces
#' the kernel matrix Ω with the symmetrized matrix
#' `Ωs = ½(Ω + a·Ω*)`, where `Ω*ₖₗ = K(xₖ, -xₗ)`.
#' The system solved is
#'
#' ```
#' [ 0   1ᵀ      ] [ b ]   [ 0 ]
#' [ 1   Ωs + YΓ ] [ α ] = [ y ]
#' ```
#'
#' where `YΓ = diag(y₁²/Γ, …, yN²/Γ)`.
#'
#' The kernel must satisfy Assumption 3 of the paper (kernel symmetry):
#' `K(-xi, xj) = K(xi, -xj)` and `K(-xi, -xj) = K(xi, xj)`.
#' RBF and even-degree polynomial kernels satisfy this; see [make_kernel()].
#'
#' @param X Numeric matrix of training inputs, one observation per row (N × p).
#' @param y Numeric vector of training targets, length N. Must satisfy `y > 0`.
#' @param kernel A kernel function created by [make_kernel()].
#' @param gamma Regularization parameter `Γ > 0`.
#' @param a Symmetry parameter: `1` for even symmetry `f(x) = f(-x)`,
#'   `-1` for odd symmetry `f(x) = -f(-x)`.
#'
#' @return An object of class `"psvr_rmspe_sym"`, a list with components:
#'   \describe{
#'     \item{`alpha`}{Numeric vector of dual variables (length N).}
#'     \item{`b`}{Numeric scalar bias term.}
#'     \item{`X_train`}{The training matrix `X` (kept for prediction).}
#'     \item{`kernel`}{The kernel function used.}
#'     \item{`gamma`}{The regularization parameter `Γ`.}
#'     \item{`a`}{The symmetry parameter.}
#'     \item{`n_train`}{Number of training observations.}
#'     \item{`p_train`}{Number of training features (columns).}
#'   }
#'
#' @examples
#' X <- matrix(c(1, 2, 3, 4, 5, 6), ncol = 2)
#' y <- c(2.1, 3.8, 6.2)
#' K <- make_kernel("rbf", sigma = 1)
#' fit <- rmspe_sym_lssvr(X, y, kernel = K, gamma = 1, a = 1)
#' predict(fit, X)
#'
#' @export
rmspe_sym_lssvr <- function(X, y, kernel, gamma, a = 1) {
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
  if (gamma <= 0)       stop("`gamma` must be positive")
  if (!a %in% c(-1, 1)) stop("`a` must be 1 (even) or -1 (odd)")

  N <- nrow(X)
  if (N > 2000L) {
    warning(sprintf(
      paste0("Large dataset (N = %d): kernel matrix is %d x %d (%.1f MB). ",
             "Consider subsampling for hyperparameter tuning."),
      N, N, N, N^2 * 8 / 1e6
    ))
  }

  Omega_s <- sym_kernel_matrix(kernel, X, a)   # ½(Ω + a·Ω*)
  diag(Omega_s) <- diag(Omega_s) + y^2 / gamma # add YΓ to diagonal in place

  # Augmented (N+1)×(N+1) system: [0, 1ᵀ; 1, Ωs+YΓ][b; α] = [0; y]
  A <- matrix(0.0, N + 1L, N + 1L)
  A[1L, 2L:(N + 1L)] <- 1.0
  A[2L:(N + 1L), 1L] <- 1.0
  A[2L:(N + 1L), 2L:(N + 1L)] <- Omega_s

  rhs <- c(0.0, y)

  sol <- solve(A, rhs)

  structure(
    list(
      alpha   = sol[2L:(N + 1L)],
      b       = sol[1L],
      X_train = X,
      kernel  = kernel,
      gamma   = gamma,
      a       = a,
      n_train = N,
      p_train = ncol(X)
    ),
    class = "psvr_rmspe_sym"
  )
}

#' Predict from a fitted symmetric LS-SVR with RMSPE model
#'
#' Prediction uses the symmetric representer:
#' `f(x) = Σₖ αₖ · ½(K(xₖ, x) + a·K(xₖ, -x)) + b`.
#'
#' @param object An object of class `"psvr_rmspe_sym"` from [rmspe_sym_lssvr()].
#' @param newdata Numeric matrix of new inputs, one observation per row (M × p).
#' @param ... Ignored.
#'
#' @return Numeric vector of length M with predicted values.
#'
#' @examples
#' X <- matrix(c(1, 2, 3, 4, 5, 6), ncol = 2)
#' y <- c(2.1, 3.8, 6.2)
#' K <- make_kernel("rbf", sigma = 1)
#' fit <- rmspe_sym_lssvr(X, y, kernel = K, gamma = 1, a = 1)
#' predict(fit, X)
#'
#' @export
predict.psvr_rmspe_sym <- function(object, newdata, ...) {
  newdata <- as.matrix(newdata)
  p <- ncol(newdata)
  if (p != object$p_train)
    stop(sprintf("newdata has %d column%s but model was trained on %d",
                 p, if (p == 1L) "" else "s", object$p_train))
  M <- nrow(newdata)
  preds <- numeric(M)
  for (i in seq_len(M)) {
    kv <- sym_kernel_vector(object$kernel, object$X_train, newdata[i, ], object$a)
    preds[i] <- sum(object$alpha * kv) + object$b
  }
  as.numeric(preds)
}

#' Print method for psvr_rmspe_sym objects
#'
#' @param x An object of class `"psvr_rmspe_sym"`.
#' @param ... Ignored.
#'
#' @return `x`, invisibly.
#'
#' @export
print.psvr_rmspe_sym <- function(x, ...) {
  ki    <- attr(x$kernel, "kernel_info")
  kdesc <- .kernel_desc(ki)
  sym   <- if (x$a == 1L) "even  (a = 1)" else "odd   (a = -1)"
  cat(sprintf(
    "\nSymmetric LS-SVR with RMSPE loss  [psvr_rmspe_sym]\n\n  Kernel:        %s\n  Gamma:         %g\n  Symmetry:      %s\n  Training obs.: %d\n\n",
    kdesc, x$gamma, sym, x$n_train
  ))
  invisible(x)
}

#' Extract coefficients from a psvr_rmspe_sym model
#'
#' @param object An object of class `"psvr_rmspe_sym"`.
#' @param ... Ignored.
#'
#' @return A named list with components:
#'   \describe{
#'     \item{`alpha`}{Dual variables / Lagrange multipliers (length N).}
#'     \item{`b`}{Bias term.}
#'     \item{`X_sv`}{Training input matrix (all N observations).}
#'   }
#'
#' @export
coef.psvr_rmspe_sym <- function(object, ...) {
  list(alpha = object$alpha, b = object$b, X_sv = object$X_train)
}
