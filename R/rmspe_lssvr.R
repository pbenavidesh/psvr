#' Fit LS-SVR with RMSPE loss (Model 3) — internal
#'
#' Internal fitter for the RMSPE LS-SVR family. Use [psvr()] with
#' `loss = "rmspe"` instead. Returns the legacy `psvr_rmspe` shape; the
#' deprecation wrapper [rmspe_lssvr()] forwards directly to this function.
#'
#' @param X,y,kernel,gamma,precondition See [rmspe_lssvr()] for the full
#'   semantics of each argument (including the Remark-17 preconditioner).
#'
#' @return A list of class `"psvr_rmspe"` (legacy shape).
#'
#' @keywords internal
.fit_rmspe <- function(X, y, kernel, gamma, precondition = "auto") {
  X <- as.matrix(X)
  y <- as.numeric(y)
  .validate_y_positive(y)
  if (gamma <= 0) stop("`gamma` must be positive")

  use_precond <- .resolve_precondition(precondition, y)

  N <- nrow(X)
  .warn_large_n(N)

  Omega <- kernel_matrix(kernel, X)
  diag(Omega) <- diag(Omega) + 1e-6           # Tikhonov jitter for PD-ness

  if (use_precond) {
    P     <- 1 / y
    Omega <- (P %o% P) * Omega                # PΩP
    diag(Omega) <- diag(Omega) + 1 / gamma    # constant-diagonal regularization
    border <- P                               # (P 1) in border / row
    rhs_y  <- P * y                           # = rep(1, N)
  } else {
    diag(Omega) <- diag(Omega) + y^2 / gamma  # add YΓ to diagonal in place
    border <- rep(1, N)                       # legacy 1ᵀ borders
    rhs_y  <- y
  }

  # Augmented (N+1)×(N+1) bordered system
  A <- matrix(0.0, N + 1L, N + 1L)
  A[1L, 2L:(N + 1L)] <- border        # top row:    [0, borderᵀ]
  A[2L:(N + 1L), 1L] <- border        # left col:   [0; border]
  A[2L:(N + 1L), 2L:(N + 1L)] <- Omega

  rhs <- c(0.0, rhs_y)

  sol <- solve(A, rhs)

  alpha <- sol[2L:(N + 1L)]
  if (use_precond) alpha <- alpha / y         # recover α = ᾱ / y

  structure(
    list(
      alpha                = alpha,
      b                    = sol[1L],
      X_train              = X,
      kernel               = kernel,
      gamma                = gamma,
      n_train              = N,
      p_train              = ncol(X),
      precondition_applied = use_precond
    ),
    class = "psvr_rmspe"
  )
}

#' Predict from a fitted LS-SVR with RMSPE model
#'
#' Method dispatched on the legacy `"psvr_rmspe"` class returned by the
#' deprecated [rmspe_lssvr()]. New code should use [psvr()].
#'
#' @param object An object of class `"psvr_rmspe"` from [rmspe_lssvr()].
#' @param newdata Numeric matrix of new inputs, one observation per row (M × p).
#' @param ... Ignored.
#'
#' @return Numeric vector of length M with predicted values.
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
    "\nLS-SVR with RMSPE loss  [psvr_rmspe]\n\n  Kernel:        %s\n  Gamma:         %g\n  Training obs.: %d\n",
    kdesc, x$gamma, x$n_train
  ))
  if (isTRUE(x$precondition_applied)) {
    cat("  Preconditioner: applied (diag(1/y) symmetric rescaling)\n")
  }
  cat("\n")
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
