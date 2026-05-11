#' Fit epsilon-SVR with MAPE loss (Model 1) — internal
#'
#' Internal fitter for the MAPE epsilon-SVR family. Use [psvr()] with
#' `loss = "mape"` instead. Returns the legacy `psvr_mape` shape; the
#' deprecation wrapper [mape_svr()] forwards directly to this function.
#'
#' @param X,y,kernel,C,eps,solver,tol See [mape_svr()].
#' @param alpha_init,alpha_star_init Optional length-N numeric warm-start
#'   vectors (Theorem 5); `NULL` cold-starts.
#' @param warm_start_check Logical; if `TRUE`, validate the post-projection
#'   feasibility of the warm-start vectors. Default `TRUE`.
#' @param new_mask Optional logical vector (length N) flagging samples that
#'   are NEW relative to the previous fit (used to distribute the equality-
#'   constraint projection over new samples only). `NULL` infers
#'   "new = both alpha and alpha_star are exactly zero".
#'
#' @return A list of class `"psvr_mape"` (legacy shape).
#'
#' @keywords internal
.fit_mape <- function(X, y, kernel, C, eps,
                      solver = c("smo", "osqp"), tol = 1e-5,
                      alpha_init = NULL,
                      alpha_star_init = NULL,
                      warm_start_check = TRUE,
                      new_mask = NULL,
                      precomputed_Omega = NULL,
                      block_k4_enabled = TRUE,
                      alpha_couple = 0.5) {
  # `precomputed_Omega` is INTERNAL — used by psvr_cv() to share a single
  # full-dataset Omega across folds. Pass the un-jittered subset
  # Omega_full[train_idx, train_idx]; this fitter adds the 1e-6 diagonal
  # jitter on the (subset of the) precomputed matrix in place.

  solver <- match.arg(solver)
  X <- as.matrix(X)
  y <- as.numeric(y)
  .validate_y_positive(y)
  if (C   <= 0) stop("`C` must be positive")
  if (eps <  0) stop("`eps` must be non-negative")

  N <- nrow(X)
  .warn_large_n(N)
  scale <- eps / 100          # ε/100, used throughout
  ub    <- 100 * C / y        # per-sample upper bounds on αk and αk*

  Omega <- if (is.null(precomputed_Omega)) {
    kernel_matrix(kernel, X)
  } else {
    stopifnot(is.matrix(precomputed_Omega),
              nrow(precomputed_Omega) == N,
              ncol(precomputed_Omega) == N)
    precomputed_Omega
  }
  diag(Omega) <- diag(Omega) + 1e-6

  iterations <- NA_integer_
  converged  <- NA
  block_k4   <- list(joint_updates               = 0L,
                     k2_fallbacks                = 0L,
                     decoupling_rate             = NA_real_,
                     early_phase_decoupling_rate = NA_real_,
                     late_phase_decoupling_rate  = NA_real_)

  if (solver == "smo") {
    K_acc      <- .make_kernel_accessor(Omega)
    sol        <- .smo_solve(K_acc, y, C, eps,
                             alpha_init = alpha_init,
                             alpha_star_init = alpha_star_init,
                             warm_start_check = warm_start_check,
                             new_mask = new_mask,
                             block_k4_enabled = block_k4_enabled,
                             alpha_couple = alpha_couple)
    alpha      <- sol$alpha
    alpha_star <- sol$alpha_star
    beta       <- alpha - alpha_star
    b          <- sol$b
    iterations <- sol$iterations
    converged  <- sol$converged
    block_k4   <- list(joint_updates               = sol$joint_updates,
                       k2_fallbacks                = sol$k2_fallbacks,
                       decoupling_rate             = sol$decoupling_rate,
                       early_phase_decoupling_rate = sol$early_phase_decoupling_rate,
                       late_phase_decoupling_rate  = sol$late_phase_decoupling_rate)
  } else {
    if (!requireNamespace("osqp", quietly = TRUE)) {
      stop('solver = "osqp" requires the osqp package. Install it with:\n',
           '  install.packages("osqp")')
    }
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
    res <- osqp::solve_osqp(P, q, A, l, u, pars = settings)

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
  }

  # ---- Retain support vectors only ----
  sv_idx <- which(abs(beta) > tol)

  structure(
    list(
      beta       = beta[sv_idx],
      alpha      = alpha,        # length-N pre-pruning (for warm-start)
      alpha_star = alpha_star,   # length-N pre-pruning (for warm-start)
      b          = b,
      X_sv       = X[sv_idx, , drop = FALSE],
      y_sv       = y[sv_idx],
      kernel     = kernel,
      C          = C,
      eps        = eps,
      n_train    = N,
      p_train    = ncol(X),
      iterations = iterations,
      converged  = converged,
      block_k4   = block_k4       # F7 telemetry
    ),
    class = "psvr_mape"
  )
}

#' Predict from a fitted epsilon-SVR with MAPE model
#'
#' Method dispatched on the legacy `"psvr_mape"` class returned by the
#' deprecated [mape_svr()]. New code should use [psvr()] (which returns a
#' `"psvr_fit"` object dispatched by [predict.psvr_fit()]).
#'
#' @param object An object of class `"psvr_mape"` from [mape_svr()].
#' @param newdata Numeric matrix of new inputs, one observation per row (M × p).
#' @param ... Ignored.
#'
#' @return Numeric vector of length M with predicted values.
#'
#' @export
predict.psvr_mape <- function(object, newdata, ...) {
  newdata <- as.matrix(newdata)
  p <- ncol(newdata)
  if (p != object$p_train)
    stop(sprintf("newdata has %d column%s but model was trained on %d",
                 p, if (p == 1L) "" else "s", object$p_train))
  M     <- nrow(newdata)
  preds <- numeric(M)
  for (i in seq_len(M)) {
    kv        <- kernel_matrix(object$kernel, object$X_sv,
                               newdata[i, , drop = FALSE])
    preds[i]  <- sum(object$beta * kv) + object$b
  }
  as.numeric(preds)
}

#' Print method for psvr_mape objects
#'
#' @param x An object of class `"psvr_mape"`.
#' @param ... Ignored.
#'
#' @return `x`, invisibly.
#'
#' @export
print.psvr_mape <- function(x, ...) {
  ki    <- attr(x$kernel, "kernel_info")
  kdesc <- .kernel_desc(ki)
  nsv   <- length(x$beta)
  cat(sprintf(
    "\nEpsilon-SVR with MAPE loss  [psvr_mape]\n\n  Kernel:          %s\n  C:               %g\n  eps:             %g\n  Training obs.:   %d\n  Support vectors: %d (%.1f%%)\n\n",
    kdesc, x$C, x$eps, x$n_train, nsv, 100 * nsv / x$n_train
  ))
  invisible(x)
}

#' Extract coefficients from a psvr_mape model
#'
#' @param object An object of class `"psvr_mape"`.
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
coef.psvr_mape <- function(object, ...) {
  list(alpha = object$beta, b = object$b, X_sv = object$X_sv)
}
