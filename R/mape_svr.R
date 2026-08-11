#' Fit epsilon-SVR with MAPE loss (Model 1) — internal
#'
#' Internal fitter for the MAPE epsilon-SVR family. Use [psvr_mape()] instead.
#' Returns the `psvr_mape` shape, which is what [psvr_mape()] and the parsnip
#' engine fit wrappers both return.
#'
#' @param X,y,kernel,C,eps,solver,tol See [psvr_mape()].
#' @param alpha_init,alpha_star_init Optional length-N numeric vectors of
#'   starting values for the two sets of SMO dual variables, typically the
#'   converged duals of a previous fit on overlapping data. They are projected
#'   back onto the constraint set before the solve (the warm-start procedure of
#'   Theorem 5 / Algorithm 1 in arXiv:2605.01446 v3). `NULL` cold-starts.
#' @param warm_start_check Logical; if `TRUE`, validate the post-projection
#'   feasibility of the warm-start vectors. Default `TRUE`.
#'
#' @return A list of class `"psvr_mape"` (legacy shape).
#'
#' @keywords internal
.fit_mape <- function(X, y, kernel, C, eps,
                      solver = c("smo", "osqp"),
                      tol = 1e-3, max_iter = 100000L,
                      alpha_init = NULL,
                      alpha_star_init = NULL,
                      warm_start_check = TRUE,
                      precomputed_Omega = NULL,
                      block_k4_enabled = TRUE,
                      alpha_couple = 0.5,
                      engine = c("rcpp", "r")) {
  engine <- match.arg(engine)
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
                             tol = tol, max_iter = max_iter,
                             alpha_init = alpha_init,
                             alpha_star_init = alpha_star_init,
                             warm_start_check = warm_start_check,
                             block_k4_enabled = block_k4_enabled,
                             alpha_couple = alpha_couple,
                             engine = engine)
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

  # ---- Training fitted values ----
  # f(xk) = Σ_{i ∈ SV} βi K(xi, xk) + b, i.e. exactly what predict() computes.
  # One matvec against the already-built Omega (no kernel rebuild); the
  # `- 1e-6 * beta_sv` term removes the diagonal jitter added at line 57, which
  # predict() does not see. beta_sv zeroes the pruned-away entries so this
  # matches the post-pruning decision function rather than the solver's
  # pre-pruning F.
  beta_sv         <- numeric(N)
  beta_sv[sv_idx] <- beta[sv_idx]
  fitted_values   <- as.numeric(Omega %*% beta_sv) - 1e-6 * beta_sv + b

  structure(
    list(
      beta       = beta[sv_idx],
      alpha      = alpha,        # length-N pre-pruning (for warm-start)
      alpha_star = alpha_star,   # length-N pre-pruning (for warm-start)
      b          = b,
      X_sv       = X[sv_idx, , drop = FALSE],
      y_sv       = y[sv_idx],
      y_train       = y,              # length N — for fitted()/residuals()
      fitted_values = fitted_values,  # length N
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
#' Method dispatched on the `"psvr_mape"` class, which both [psvr_mape()] and
#' the parsnip engine fit wrappers return.
#'
#' @param object An object of class `"psvr_mape"`, as returned by [psvr_mape()]
#'   with `sym_type = "none"` or by the parsnip engine fit wrappers (see
#'   [psvr-fit-wrappers]; `sym_type = "even"` or `"odd"` yields
#'   `"psvr_mape_sym"` instead). Unwrap a parsnip fit with
#'   [parsnip::extract_fit_engine()] to obtain it.
#' @param newdata Numeric matrix of new inputs, one observation per row
#'   (\eqn{M \times p}{M x p}).
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
  Kb <- kernel_matrix(object$kernel, object$X_sv, newdata)
  as.numeric(colSums(object$beta * Kb) + object$b)
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
#' @return A named list with five components:
#'   \describe{
#'     \item{`alpha`, `alpha_star`}{The length-`N` pre-pruning dual variables
#'       \eqn{\alpha_k}{alpha_k} and \eqn{\alpha^*_k}{alpha_star_k}.}
#'     \item{`beta`}{The pruned dual differences
#'       \eqn{\beta_k = \alpha_k - \alpha^*_k}{beta_k = alpha_k - alpha_star_k}
#'       over the support-vector indices (length `n_sv`); this is what
#'       `predict()` uses.}
#'     \item{`b`}{Bias term.}
#'     \item{`support_data`}{Support vector input matrix.}
#'   }
#'
#'   The LS-SVR classes return **three** components rather than five, since
#'   they have no `alpha_star` and no pruned `beta`; the absent components are
#'   not materialised as `NULL`. So `names(coef(fit))` depends on the model
#'   family. That is deliberate: each class is family-specific, and inventing
#'   empty slots to make the two agree would add structure with nothing to
#'   inherit it from.
#'
#' @section Renamed in 0.0.2.9011:
#' `alpha` previously held the pruned \eqn{\beta}{beta} and `support_data` was
#' named `X_sv`, which made `coef(fit)$alpha` mean the length-`n_sv`
#' \eqn{\beta}{beta} here but the length-`N` dual \eqn{\alpha}{alpha} on a fit
#' from the superseded `psvr()`: one generic returning two different vectors under
#' one name, silently, depending on entry point. The
#' \eqn{\beta}{beta}-under-`alpha` meaning is the one 0.0.2.9004 moved away
#' from on the object itself; this aligns `coef()` with it.
#'
#' @export
coef.psvr_mape <- function(object, ...) {
  list(alpha        = object$alpha,        # length N, pre-pruning
       alpha_star   = object$alpha_star,   # length N, pre-pruning
       beta         = object$beta,         # length n_sv, used by predict()
       b            = object$b,
       support_data = object$X_sv)
}
