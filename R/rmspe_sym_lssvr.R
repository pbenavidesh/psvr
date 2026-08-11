#' Fit symmetric LS-SVR with RMSPE loss (Model 4) — internal
#'
#' Internal fitter for the symmetric RMSPE LS-SVR family. Use [psvr_rmspe()]
#' with `sym_type = "even"` / `"odd"` instead. Returns the `psvr_rmspe_sym`
#' shape, which is what [psvr_rmspe()] and the parsnip engine fit wrappers both
#' return with `sym_type = "even"` / `"odd"`. The kernel must
#' satisfy Assumption 3 of the paper (kernel symmetry); see [make_kernel()].
#'
#' @param X,y,kernel,gamma,precondition See [psvr_rmspe()] for the full
#'   semantics of each argument, including the `diag(1/y)` preconditioner that
#'   flattens the target-weighted diagonal to a constant before the solve.
#' @param a Symmetry type: `1` (even) or `-1` (odd). This is the internal
#'   integer; the public argument is `sym_type`, with `"even"` mapping to
#'   `a = 1` and `"odd"` to `a = -1`. Neither [psvr_rmspe()] nor the parsnip
#'   specifications expose `a` directly.
#'
#' @return A list of class `"psvr_rmspe_sym"`.
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

  # ---- Training fitted values ----
  # Same KKT identity as Model 3 with Ωs in place of Ω: the solved system is
  # (Ωs + 1e-6·I + YΓ)α + b·1 = y, so f(xk) = (Ωs α)[k] + b
  # = yk - (1e-6 + yk²/Γ)·αk. O(N); `Omega_s` has been destructively modified
  # above and is no longer the matrix predict() uses.
  fitted_values <- y - (1e-6 + y^2 / gamma) * alpha

  structure(
    list(
      alpha                = alpha,
      b                    = sol[1L],
      X_train              = X,
      y_train              = y,              # length N — for fitted()/residuals()
      fitted_values        = fitted_values,  # length N
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
#' Method dispatched on the `"psvr_rmspe_sym"` class, which both [psvr_rmspe()]
#' and the parsnip engine fit wrappers return. Uses the symmetric representer
#' \deqn{f(x) = \sum_k \alpha_k \cdot
#'              \tfrac{1}{2}\left(K(x_k, x) + a K(x_k, -x)\right) + b}{%
#'       f(x) = sum_k alpha_k * 0.5 * (K(x_k, x) + a * K(x_k, -x)) + b}
#'
#' @param object An object of class `"psvr_rmspe_sym"`, as returned by
#'   [psvr_rmspe()] with `sym_type = "even"` or `"odd"`, or by the parsnip
#'   engine fit wrappers (see [psvr-fit-wrappers]; `sym_type = "none"` yields
#'   `"psvr_rmspe"` instead). Unwrap a parsnip fit with
#'   [parsnip::extract_fit_engine()] to obtain it.
#' @param newdata Numeric matrix of new inputs, one observation per row
#'   (\eqn{M \times p}{M x p}).
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
  Kb <- sym_kernel_block(object$kernel, object$X_train, newdata, object$a)
  as.numeric(colSums(object$alpha * Kb) + object$b)
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
#'     \item{`support_data`}{Training input matrix (all N observations).}
#'   }
#'   Three components, not the five the MAPE classes return: LS-SVR has no
#'   `alpha_star` and no pruned `beta`, and they are not materialised as
#'   `NULL`. See [coef.psvr_rmspe()].
#'
#' @section Renamed in 0.0.2.9011:
#' See [coef.psvr_rmspe()] — the same rename, for the same reason, applied to
#' both LS-SVR classes together.
#'
#' @export
coef.psvr_rmspe_sym <- function(object, ...) {
  list(alpha = object$alpha, b = object$b, support_data = object$X_train)
}
