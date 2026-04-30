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
#' @param precondition Optional symmetric rescaling preconditioner derived
#'   from Remark 17 of the companion paper, used when the target dynamic range
#'   `ρ = max(y) / min(y)` is large enough to make `Ω + YΓ` ill-conditioned.
#'   One of:
#'   \describe{
#'     \item{`"auto"` (default)}{Apply the preconditioner when `ρ > 10`.}
#'     \item{`"always"`}{Apply the preconditioner unconditionally.}
#'     \item{`"never"`}{Disable the preconditioner (legacy behaviour).}
#'     \item{a positive numeric scalar}{Apply when `ρ > precondition`.}
#'   }
#'
#' @details
#' When the target dynamic range `ρ = max(y) / min(y)` is large, the
#' diagonal of `YΓ = diag(yₖ²/Γ)` varies as `O(ρ²)`, making `Ω + YΓ`
#' ill-conditioned. Remark 17 of the companion paper derives a symmetric
#' rescaling preconditioner `P = diag(1/yₖ)` via the change of variable
#' `α = P ᾱ` (i.e. `αₖ = ᾱₖ / yₖ`). Multiplying the inner block of the
#' bordered system by `P` from the left gives
#' `(P Ω P + Γ⁻¹·I) ᾱ = P y − b · P 1 = 1 − b · P 1`,
#' with constant-diagonal regularization `Γ⁻¹ · I` independent of `yₖ`.
#' The constraint `1ᵀ α = 0` becomes `(P 1)ᵀ ᾱ = 0`, so the bordered
#' system used by the preconditioned solver is
#' \preformatted{
#' [ 0      (P 1)ᵀ          ] [ b ]   [ 0 ]
#' [ P 1    P Ω P + Γ⁻¹·I    ] [ ᾱ ] = [ 1 ]
#' }
#' Recovery is `α = ᾱ / y` (elementwise division). The bias `b` is the
#' same constraint multiplier in both systems.
#'
#' This is a strict change of variable: in exact arithmetic the
#' preconditioned and unconditioned solvers produce identical predictions.
#' Its purpose is to preserve solver accuracy under finite floating-point
#' precision when `ρ` is large; for moderate `ρ` the two paths agree to
#' within machine epsilon. Use `precondition = "auto"` (default) for
#' typical workloads, `"never"` for legacy behaviour, or a custom numeric
#' threshold for fine-grained control. The chosen behaviour is recorded
#' in `precondition_applied`.
#'
#' @return An object of class `"psvr_rmspe"`, a list with components:
#'   \describe{
#'     \item{`alpha`}{Numeric vector of dual variables (length N), in the
#'       original variable space.}
#'     \item{`b`}{Numeric scalar bias term.}
#'     \item{`X_train`}{The training matrix `X` (kept for prediction).}
#'     \item{`kernel`}{The kernel function used.}
#'     \item{`gamma`}{The regularization parameter `Γ`.}
#'     \item{`n_train`}{Number of training observations.}
#'     \item{`p_train`}{Number of training features (columns).}
#'     \item{`precondition_applied`}{Logical scalar; `TRUE` if the
#'       preconditioner was applied for this fit.}
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
rmspe_lssvr <- function(X, y, kernel, gamma, precondition = "auto") {
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

  use_precond <- .resolve_precondition(precondition, y)

  N <- nrow(X)
  if (N > 2000L) {
    warning(sprintf(
      paste0("Large dataset (N = %d): kernel matrix is %d x %d (%.1f MB). ",
             "Consider subsampling for hyperparameter tuning."),
      N, N, N, N^2 * 8 / 1e6
    ))
  }

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

# Resolve the `precondition` argument shared by the LS-SVR fitters.
# Returns a single logical (TRUE = apply, FALSE = don't).
.resolve_precondition <- function(precondition, y) {
  if (is.character(precondition)) {
    if (length(precondition) != 1L)
      stop("`precondition` must be a length-1 character or numeric scalar")
    choices <- c("always", "never", "auto")
    if (!precondition %in% choices)
      stop(sprintf(
        "`precondition` must be one of %s, or a positive numeric threshold",
        paste(sprintf('"%s"', choices), collapse = ", ")
      ))
    switch(
      precondition,
      always = TRUE,
      never  = FALSE,
      auto   = (max(y) / min(y)) > 10
    )
  } else if (is.numeric(precondition)) {
    if (length(precondition) != 1L || !is.finite(precondition) || precondition <= 0)
      stop("`precondition` numeric threshold must be a single positive finite value")
    (max(y) / min(y)) > as.numeric(precondition)
  } else {
    stop('`precondition` must be one of "always", "never", "auto", or a positive numeric threshold')
  }
}
