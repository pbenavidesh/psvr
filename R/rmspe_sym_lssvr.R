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
#' @param precondition Optional symmetric rescaling preconditioner derived
#'   from Remark 17 of the companion paper, used when the target dynamic range
#'   `ρ = max(y) / min(y)` is large enough to make `Ωs + YΓ` ill-conditioned.
#'   One of:
#'   \describe{
#'     \item{`"auto"` (default)}{Apply the preconditioner when `ρ > 10`.}
#'     \item{`"always"`}{Apply the preconditioner unconditionally.}
#'     \item{`"never"`}{Disable the preconditioner (legacy behaviour).}
#'     \item{a positive numeric scalar}{Apply when `ρ > precondition`.}
#'   }
#'
#' @details
#' When `ρ = max(y) / min(y)` is large, `Ωs + YΓ` becomes ill-conditioned
#' because the diagonal of `YΓ = diag(yₖ²/Γ)` varies as `O(ρ²)`. Remark 17
#' of the companion paper derives a symmetric rescaling preconditioner
#' `P = diag(1/yₖ)` via the change of variable `α = P ᾱ`
#' (i.e. `αₖ = ᾱₖ / yₖ`). Multiplying the inner block of the bordered
#' system by `P` from the left gives
#' `(P Ωs P + Γ⁻¹·I) ᾱ = P y − b · P 1 = 1 − b · P 1`,
#' with constant-diagonal regularization `Γ⁻¹ · I`. The preconditioner is
#' applied to the symmetrized kernel matrix `Ωs` (after symmetrization).
#' The constraint `1ᵀ α = 0` becomes `(P 1)ᵀ ᾱ = 0`, so the bordered
#' system used by the preconditioned solver is
#' \preformatted{
#' [ 0      (P 1)ᵀ          ] [ b ]   [ 0 ]
#' [ P 1    P Ωs P + Γ⁻¹·I   ] [ ᾱ ] = [ 1 ]
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
#' @return An object of class `"psvr_rmspe_sym"`, a list with components:
#'   \describe{
#'     \item{`alpha`}{Numeric vector of dual variables (length N), in the
#'       original variable space.}
#'     \item{`b`}{Numeric scalar bias term.}
#'     \item{`X_train`}{The training matrix `X` (kept for prediction).}
#'     \item{`kernel`}{The kernel function used.}
#'     \item{`gamma`}{The regularization parameter `Γ`.}
#'     \item{`a`}{The symmetry parameter.}
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
#' fit <- rmspe_sym_lssvr(X, y, kernel = K, gamma = 1, a = 1)
#' predict(fit, X)
#'
#' @export
rmspe_sym_lssvr <- function(X, y, kernel, gamma, a = 1, precondition = "auto") {
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
