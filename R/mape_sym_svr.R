#' Fit symmetric epsilon-SVR with MAPE loss (Model 2) — internal
#'
#' Internal fitter for the symmetric MAPE epsilon-SVR family. Use [psvr()]
#' with `loss = "mape"` and `sym = +1L` / `-1L` instead. Returns the legacy
#' `psvr_mape_sym` shape; the deprecation wrapper [mape_sym_svr()] forwards
#' directly to this function. The kernel must satisfy Assumption 3 of the
#' paper; see [make_kernel()].
#'
#' @param X,y,kernel,C,eps,a,solver,tol See [mape_sym_svr()].
#'
#' @return A list of class `"psvr_mape_sym"` (legacy shape).
#'
#' @keywords internal
.fit_mape_sym <- function(X, y, kernel, C, eps, a = 1,
                          solver = c("smo", "osqp"), tol = 1e-5) {
  solver <- match.arg(solver)
  X <- as.matrix(X)
  y <- as.numeric(y)
  .validate_y_positive(y)
  if (C   <= 0)           stop("`C` must be positive")
  if (eps <  0)           stop("`eps` must be non-negative")
  if (!a %in% c(-1L, 1L)) stop("`a` must be 1 (even) or -1 (odd)")

  N <- nrow(X)
  .warn_large_n(N)
  scale <- eps / 100
  ub    <- 100 * C / y

  # Ωs = ½(Ω + a·Ω*) — symmetric kernel matrix with the representer-theorem
  # ½ already baked in.  Same convention used by Model 4.
  Omega_s <- sym_kernel_matrix(kernel, X, a)
  # 0.5e-6 (not 1e-6) preserves bit-identicality with pre-F1 behavior, where
  # diag(Ks) += 1e-6 was followed by 0.5*Ks, giving an effective Ωs jitter
  # of 0.5e-6.
  diag(Omega_s) <- diag(Omega_s) + 0.5e-6

  if (solver == "smo") {
    # Pass Ωs directly; matches Model 1's SMO contract (the ½ from the
    # symmetric representer is already in Ωs).  Bias `b` and bounds match
    # Model 1.
    K_acc      <- .make_kernel_accessor(Omega_s)
    sol        <- .smo_solve(K_acc, y, C, eps)
    alpha      <- sol$alpha
    alpha_star <- sol$alpha_star
    beta       <- alpha - alpha_star
    b          <- sol$b
  } else {
    if (!requireNamespace("osqp", quietly = TRUE)) {
      stop('solver = "osqp" requires the osqp package. Install it with:\n',
           '  install.packages("osqp")')
    }
    # ---- QP matrices ----
    # P = [Ωs,-Ωs;-Ωs,Ωs] so that ½uᵀPu = ½βᵀΩsβ = ¼βᵀKsβ  (Theorem 2)
    P_dense <- rbind(cbind(Omega_s, -Omega_s), cbind(-Omega_s, Omega_s))
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
    # Symmetric representer: f(xk) = (Ωs·β)[k] + b   (Ωs = ½Ks)
    Omega_s_beta <- as.numeric(Omega_s %*% beta)

    free_up <- which(alpha      > tol & alpha      < ub - tol)
    free_lo <- which(alpha_star > tol & alpha_star < ub - tol)

    # Free upper: yk - f(xk) = ε·yk/100  →  b = yk(1-ε/100) - (Ωs·β)[k]
    # Free lower: f(xk) - yk = ε·yk/100  →  b = yk(1+ε/100) - (Ωs·β)[k]
    b_up <- y[free_up] * (1.0 - scale) - Omega_s_beta[free_up]
    b_lo <- y[free_lo] * (1.0 + scale) - Omega_s_beta[free_lo]

    b_vals <- c(b_up, b_lo)

    if (length(b_vals) == 0L) {
      sat_up    <- which(alpha      > tol)
      sat_lo    <- which(alpha_star > tol)
      bub_bound <- if (length(sat_up) > 0L) min(y[sat_up] * (1.0 - scale) - Omega_s_beta[sat_up]) else  Inf
      blb_bound <- if (length(sat_lo) > 0L) max(y[sat_lo] * (1.0 + scale) - Omega_s_beta[sat_lo]) else -Inf
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
#' Method dispatched on the legacy `"psvr_mape_sym"` class returned by the
#' deprecated [mape_sym_svr()]. Uses the symmetric representer theorem
#' `f(x) = ½ Σk βk Ks(xk, x) + b` with
#' `Ks(xk, x) = K(xk, x) + a·K(xk, -x)`. New code should use [psvr()].
#'
#' @param object An object of class `"psvr_mape_sym"` from [mape_sym_svr()].
#' @param newdata Numeric matrix of new inputs, one observation per row (M × p).
#' @param ... Ignored.
#'
#' @return Numeric vector of length M with predicted values.
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
