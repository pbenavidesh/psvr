#' Fit LS-SVR with RMSPE loss (Model 3)
#'
#' Solves the linear system derived in Theorem 3 of Benavides-Herrera et al.
#' (2026). The primal objective is
#' `½‖ω‖² + (Γ/2) Σ eₖ²/yₖ²`, leading to the (N+1)×(N+1) system
#'
#' ```
#' [ 0   1ᵀ     ] [ b ]   [ 0 ]
#' [ 1   Ω + YΓ ] [ α ] = [ y ]
#' ```
#'
#' where `YΓ = diag(y₁²/Γ, …, yN²/Γ)` is added to the diagonal of Ω.
#'
#' @param X Numeric matrix of training inputs, one observation per row (N × p).
#' @param y Numeric vector of training targets, length N. Must satisfy `y > 0`.
#' @param kernel A kernel function created by [make_kernel()].
#' @param gamma Regularization parameter `Γ > 0`.
#'
#' @return An object of class `"psvr_rmspe"`, a list with components:
#'   \describe{
#'     \item{`alpha`}{Numeric vector of dual variables (length N).}
#'     \item{`b`}{Numeric scalar bias term.}
#'     \item{`X_train`}{The training matrix `X` (kept for prediction).}
#'     \item{`kernel`}{The kernel function used.}
#'     \item{`gamma`}{The regularization parameter `Γ`.}
#'     \item{`n_train`}{Number of training observations.}
#'     \item{`p_train`}{Number of training features (columns).}
#'   }
#'
#' @examples
#' X <- matrix(c(1, 2, 3, 4, 5, 6), ncol = 2)
#' y <- c(2.1, 3.8, 6.2)
#' K <- make_kernel("rbf", sigma = 1)
#' fit <- rmspe_lssvr(X, y, kernel = K, gamma = 1)
#' predict(fit, X)
#'
#' @export
rmspe_lssvr <- function(X, y, kernel, gamma) {
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
  if (gamma <= 0)  stop("`gamma` must be positive")

  N <- nrow(X)
  if (N > 2000L) {
    warning(sprintf(
      paste0("Large dataset (N = %d): kernel matrix is %d x %d (%.1f MB). ",
             "Consider subsampling for hyperparameter tuning."),
      N, N, N, N^2 * 8 / 1e6
    ))
  }

  Omega <- kernel_matrix(kernel, X)
  diag(Omega) <- diag(Omega) + y^2 / gamma   # add YΓ to diagonal in place

  # Augmented (N+1)×(N+1) system: [0, 1ᵀ; 1, Ω+YΓ][b; α] = [0; y]
  A <- matrix(0.0, N + 1L, N + 1L)
  A[1L, 2L:(N + 1L)] <- 1.0           # top row:    [0, 1ᵀ]
  A[2L:(N + 1L), 1L] <- 1.0           # left col:   [0; 1]
  A[2L:(N + 1L), 2L:(N + 1L)] <- Omega

  rhs <- c(0.0, y)

  sol <- solve(A, rhs)

  structure(
    list(
      alpha   = sol[2L:(N + 1L)],
      b       = sol[1L],
      X_train = X,
      kernel  = kernel,
      gamma   = gamma,
      n_train = N,
      p_train = ncol(X)
    ),
    class = "psvr_rmspe"
  )
}

#' Predict from a fitted LS-SVR with RMSPE model
#'
#' @param object An object of class `"psvr_rmspe"` from [rmspe_lssvr()].
#' @param newdata Numeric matrix of new inputs, one observation per row (M × p).
#' @param ... Ignored.
#'
#' @return Numeric vector of length M with predicted values.
#'
#' @examples
#' X <- matrix(c(1, 2, 3, 4, 5, 6), ncol = 2)
#' y <- c(2.1, 3.8, 6.2)
#' K <- make_kernel("rbf", sigma = 1)
#' fit <- rmspe_lssvr(X, y, kernel = K, gamma = 1)
#' predict(fit, X)
#'
#' @export
predict.psvr_rmspe <- function(object, newdata, ...) {
  newdata <- as.matrix(newdata)
  p <- ncol(newdata)
  if (p != object$p_train)
    stop(sprintf("newdata has %d column%s but model was trained on %d",
                 p, if (p == 1L) "" else "s", object$p_train))
  M <- nrow(newdata)
  preds <- numeric(M)
  for (i in seq_len(M)) {
    kv <- kernel_matrix(object$kernel, object$X_train, newdata[i, , drop = FALSE])
    preds[i] <- sum(object$alpha * kv) + object$b
  }
  as.numeric(preds)
}

#' Print method for psvr_rmspe objects
#'
#' @param x An object of class `"psvr_rmspe"`.
#' @param ... Ignored.
#'
#' @return `x`, invisibly.
#'
#' @export
print.psvr_rmspe <- function(x, ...) {
  ki    <- attr(x$kernel, "kernel_info")
  kdesc <- .kernel_desc(ki)
  cat(sprintf(
    "\nLS-SVR with RMSPE loss  [psvr_rmspe]\n\n  Kernel:        %s\n  Gamma:         %g\n  Training obs.: %d\n\n",
    kdesc, x$gamma, x$n_train
  ))
  invisible(x)
}

#' Extract coefficients from a psvr_rmspe model
#'
#' @param object An object of class `"psvr_rmspe"`.
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
coef.psvr_rmspe <- function(object, ...) {
  list(alpha = object$alpha, b = object$b, X_sv = object$X_train)
}
