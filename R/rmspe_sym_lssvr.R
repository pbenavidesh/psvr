#' Fit symmetric LS-SVR with RMSPE loss (Model 4) — internal
#'
#' Internal fitter for the symmetric RMSPE LS-SVR family. Use [psvr()]
#' with `loss = "rmspe"` and `sym = +1L` / `-1L` instead. Returns the
#' legacy `psvr_rmspe_sym` shape; the deprecation wrapper
#' [rmspe_sym_lssvr()] forwards directly to this function. The kernel must
#' satisfy Assumption 3 of the paper (kernel symmetry); see [make_kernel()].
#'
#' @param X,y,kernel,gamma,a,precondition See [rmspe_sym_lssvr()] for the
#'   full semantics of each argument (including the Remark-17
#'   preconditioner).
#'
#' @return A list of class `"psvr_rmspe_sym"` (legacy shape).
#'
#' @keywords internal
.fit_rmspe_sym <- function(X, y, kernel, gamma, a = 1, precondition = "auto") {
  X <- as.matrix(X)
  y <- as.numeric(y)
  .validate_y_positive(y)
  if (gamma <= 0)       stop("`gamma` must be positive")
  if (!a %in% c(-1, 1)) stop("`a` must be 1 (even) or -1 (odd)")

  use_precond <- .resolve_precondition(precondition, y)

  N <- nrow(X)
  .warn_large_n(N)

  Omega_s <- sym_kernel_matrix(kernel, X, a)   # ½(Ω + a·Ω*)
  diag(Omega_s) <- diag(Omega_s) + 1e-6        # Tikhonov jitter

  if (use_precond) {
    P       <- 1 / y
    Omega_s <- (P %o% P) * Omega_s             # P Ωs P
    diag(Omega_s) <- diag(Omega_s) + 1 / gamma
    border <- P                                # (P 1) in border / row
    rhs_y  <- P * y                            # = rep(1, N)
  } else {
    diag(Omega_s) <- diag(Omega_s) + y^2 / gamma  # add YΓ to diagonal
    border <- rep(1, N)                        # legacy 1ᵀ borders
    rhs_y  <- y
  }

  # Augmented (N+1)×(N+1) bordered system
  A <- matrix(0.0, N + 1L, N + 1L)
  A[1L, 2L:(N + 1L)] <- border
  A[2L:(N + 1L), 1L] <- border
  A[2L:(N + 1L), 2L:(N + 1L)] <- Omega_s

  rhs <- c(0.0, rhs_y)

  sol <- solve(A, rhs)

  alpha <- sol[2L:(N + 1L)]
  if (use_precond) alpha <- alpha / y          # recover α = ᾱ / y

  structure(
    list(
      alpha                = alpha,
      b                    = sol[1L],
      X_train              = X,
      kernel               = kernel,
      gamma                = gamma,
      a                    = a,
      n_train              = N,
      p_train              = ncol(X),
      precondition_applied = use_precond
    ),
    class = "psvr_rmspe_sym"
  )
}

#' Predict from a fitted symmetric LS-SVR with RMSPE model
#'
#' Method dispatched on the legacy `"psvr_rmspe_sym"` class returned by the
#' deprecated [rmspe_sym_lssvr()]. Uses the symmetric representer
#' `f(x) = Σₖ αₖ · ½(K(xₖ, x) + a·K(xₖ, -x)) + b`. New code should use
#' [psvr()].
#'
#' @param object An object of class `"psvr_rmspe_sym"` from [rmspe_sym_lssvr()].
#' @param newdata Numeric matrix of new inputs, one observation per row (M × p).
#' @param ... Ignored.
#'
#' @return Numeric vector of length M with predicted values.
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
    "\nSymmetric LS-SVR with RMSPE loss  [psvr_rmspe_sym]\n\n  Kernel:        %s\n  Gamma:         %g\n  Symmetry:      %s\n  Training obs.: %d\n",
    kdesc, x$gamma, sym, x$n_train
  ))
  if (isTRUE(x$precondition_applied)) {
    cat("  Preconditioner: applied (diag(1/y) symmetric rescaling)\n")
  }
  cat("\n")
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
