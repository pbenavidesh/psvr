#' Predict from a fitted psvr_fit model
#'
#' @param object An object of class `"psvr_fit"` from [psvr()].
#' @param newdata Numeric matrix of new inputs, one observation per row (M × p).
#' @param ... Ignored.
#'
#' @return Numeric vector of length M with predicted values.
#'
#' @examples
#' set.seed(1)
#' X <- matrix(rnorm(40), 20, 2)
#' y <- rlnorm(20)
#' fit <- psvr(X, y, loss = "rmspe", kernel = make_kernel("rbf", sigma = 1),
#'             gamma = 100)
#' predict(fit, X[1:3, , drop = FALSE])
#'
#' @export
predict.psvr_fit <- function(object, newdata, ...) {
  .psvr_predict_dispatch(object, newdata)
}

#' Print method for psvr_fit objects
#'
#' @param x An object of class `"psvr_fit"`.
#' @param ... Ignored.
#'
#' @return `x`, invisibly.
#'
#' @export
print.psvr_fit <- function(x, ...) {
  ki         <- attr(x$kernel, "kernel_info")
  kdesc      <- .kernel_desc(ki)
  is_mape    <- x$loss == "mape"
  is_sym     <- !is.null(x$sym)
  loss_label <- if (is_mape) "epsilon-SVR with MAPE loss" else "LS-SVR with RMSPE loss"
  sym_prefix <- if (is_sym) "Symmetric " else ""

  cat(sprintf("\n%s%s  [psvr_fit]\n\n", sym_prefix, loss_label))
  cat(sprintf("  Kernel:          %s\n", kdesc))

  hp <- x$hyperparameters
  if (is_mape) {
    cat(sprintf("  C:               %g\n", hp$C))
    cat(sprintf("  eps:             %g\n", hp$eps))
  } else {
    cat(sprintf("  Gamma:           %g\n", hp$gamma))
  }
  if (is_sym) {
    sym_label <- if (x$sym == 1L) "even  (a = 1)" else "odd   (a = -1)"
    cat(sprintf("  Symmetry:        %s\n", sym_label))
  }
  cat(sprintf("  Training obs.:   %d\n", x$n_train))
  if (is_mape) {
    cat(sprintf("  Support vectors: %d (%.1f%%)\n",
                x$n_sv, 100 * x$n_sv / x$n_train))
  }
  if (isTRUE(x$solver_meta$precondition_applied)) {
    cat("  Preconditioner:  applied (diag(1/y) symmetric rescaling)\n")
  }
  cat("\n")
  invisible(x)
}

#' Extract coefficients from a psvr_fit model
#'
#' @param object An object of class `"psvr_fit"`.
#' @param ... Ignored.
#'
#' @return A named list. For `loss = "mape"` it contains `alpha` and
#'   `alpha_star` (the length-`N` pre-pruning dual variables), `beta`
#'   (the pruned `α − α*` of length `n_sv` used by `predict()`), `b`,
#'   and `support_data`. For `loss = "rmspe"`, `alpha` (length `N`,
#'   the LS-SVR solution), `b`, and `support_data` (with `alpha_star`
#'   and `beta` set to `NULL`).
#'
#' @export
coef.psvr_fit <- function(object, ...) {
  list(alpha        = object$alpha,
       alpha_star   = object$alpha_star,
       beta         = object$beta,
       b            = object$b,
       support_data = object$support_data)
}

#' Summary method for psvr_fit objects
#'
#' @param object An object of class `"psvr_fit"`.
#' @param ... Ignored.
#'
#' @return `object`, invisibly. Side effect: prints a multi-line summary.
#'
#' @export
summary.psvr_fit <- function(object, ...) {
  ki    <- attr(object$kernel, "kernel_info")
  kdesc <- .kernel_desc(ki)
  sym_label <- if (is.null(object$sym))   "none"
               else if (object$sym == 1L) "+1 (even)"
               else                        "-1 (odd)"

  cat(sprintf("\npsvr_fit  [loss = %s, sym = %s]\n\n",
              object$loss, sym_label))
  cat(sprintf("  Kernel:          %s\n", kdesc))
  cat(sprintf("  Training obs.:   %d\n", object$n_train))
  cat(sprintf("  Support vectors: %d (%.1f%%)\n",
              object$n_sv, 100 * object$n_sv / object$n_train))

  cat("\n  Hyperparameters:\n")
  hp <- object$hyperparameters
  for (nm in names(hp)) {
    if (!is.null(hp[[nm]]))
      cat(sprintf("    %-6s = %g\n", nm, hp[[nm]]))
  }

  cat("\n  Solver: ", object$solver_meta$backend, "\n", sep = "")
  if (isTRUE(object$solver_meta$precondition_applied))
    cat("  Preconditioner: applied (diag(1/y) symmetric rescaling)\n")
  cat("\n")
  invisible(object)
}
