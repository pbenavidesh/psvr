# summary() for the four fit classes.
#
# Until API-redesign stage 5 the generic was registered on psvr_fit only, so it
# was unreachable from parsnip::extract_fit_engine(). One method per class, in
# the same one-method-per-class style as predict/print/coef.
#
# Each is shorter than the psvr_fit method it replaces, for a structural
# reason: that one served both families from a single class, so it had to skip
# NULL hyperparameters in a loop and guard `precondition_applied` with
# isTRUE(). A family-specific method knows which fields exist.
#
# Deliberately NOT printed: a "Solver: <backend>" line. No fit class records
# which backend ran -- `solver` is a fitter argument, not a stored field --
# and adding one would be shape drift beyond the split. The MAPE methods print
# `iterations` and `converged` instead, which is strictly more informative than
# the constant string would have been.
#
# ASCII-only, including the roxygen -- see the note at the head of
# R/psvr_mape.R.

# Shared body for the two epsilon-SVR classes. `a` is NULL for psvr_mape.
.summary_mape <- function(object, cls) {
  kdesc <- .kernel_desc(attr(object$kernel, "kernel_info"))
  n_sv  <- length(object$beta)

  cat(sprintf("\nEpsilon-SVR with MAPE loss  [%s]\n\n", cls))
  cat(sprintf("  Kernel:          %s\n", kdesc))
  cat(sprintf("  Training obs.:   %d\n", object$n_train))
  cat(sprintf("  Predictors:      %d\n", object$p_train))
  cat(sprintf("  Support vectors: %d (%.1f%%)\n",
              n_sv, 100 * n_sv / object$n_train))
  # `"a" %in% names(object)`, NOT `!is.null(object$a)`: `$` does partial
  # matching on lists, and every one of these classes has `alpha`. On
  # psvr_rmspe, `object$a` matches `alpha` uniquely and returns the length-N
  # multiplier vector, so the branch below would be handed a vector. (On
  # psvr_mape the same expression happens to yield NULL only because `alpha`
  # and `alpha_star` make the prefix ambiguous -- correct by luck, which is
  # worse than wrong.)
  if ("a" %in% names(object))
    cat(sprintf("  Symmetry:        %s\n",
                if (object$a == 1L) "even  (a = 1)" else "odd   (a = -1)"))

  cat("\n  Hyperparameters:\n")
  cat(sprintf("    C      = %g\n", object$C))
  cat(sprintf("    eps    = %g\n", object$eps))

  cat(sprintf("\n  SMO iterations:  %s%s\n",
              format(object$iterations),
              if (isTRUE(object$converged)) " (converged)"
              else " (DID NOT CONVERGE)"))
  cat("\n")
  invisible(object)
}

# Shared body for the two LS-SVR classes. `a` is NULL for psvr_rmspe.
.summary_rmspe <- function(object, cls) {
  kdesc <- .kernel_desc(attr(object$kernel, "kernel_info"))

  cat(sprintf("\nLS-SVR with RMSPE loss  [%s]\n\n", cls))
  cat(sprintf("  Kernel:          %s\n", kdesc))
  cat(sprintf("  Training obs.:   %d\n", object$n_train))
  cat(sprintf("  Predictors:      %d\n", object$p_train))
  # `"a" %in% names(object)`, NOT `!is.null(object$a)`: `$` does partial
  # matching on lists, and every one of these classes has `alpha`. On
  # psvr_rmspe, `object$a` matches `alpha` uniquely and returns the length-N
  # multiplier vector, so the branch below would be handed a vector. (On
  # psvr_mape the same expression happens to yield NULL only because `alpha`
  # and `alpha_star` make the prefix ambiguous -- correct by luck, which is
  # worse than wrong.)
  if ("a" %in% names(object))
    cat(sprintf("  Symmetry:        %s\n",
                if (object$a == 1L) "even  (a = 1)" else "odd   (a = -1)"))

  cat("\n  Hyperparameters:\n")
  cat(sprintf("    Gamma  = %g\n", object$gamma))

  cat(sprintf("\n  Preconditioner:  %s\n",
              if (isTRUE(object$precondition_applied))
                "applied (diag(1/y) symmetric rescaling)"
              else "not applied"))
  cat("\n")
  invisible(object)
}

#' Summarize a fitted epsilon-SVR with MAPE loss
#'
#' Prints the kernel, the training and support-vector counts, the
#' hyperparameters, and the SMO iteration count with its convergence status.
#' Every training point contributes for LS-SVR but not here: the support-vector
#' percentage is the sparsity of the fit.
#'
#' @param object An object of class `"psvr_mape"`, from [psvr_mape()] with
#'   `sym_type = "none"`, or from a parsnip fit unwrapped with
#'   [parsnip::extract_fit_engine()].
#' @param ... Ignored.
#'
#' @return `object`, invisibly. Called for the printed summary.
#'
#' @seealso [print.psvr_mape()], [coef.psvr_mape()]
#'
#' @examples
#' set.seed(1)
#' X <- matrix(rnorm(40), 20, 2)
#' y <- rlnorm(20)
#' fit <- psvr_mape(X, y, kernel = make_kernel("rbf", sigma = 1),
#'                  C = 10, eps = 5)
#' summary(fit)
#'
#' @export
summary.psvr_mape <- function(object, ...) .summary_mape(object, "psvr_mape")

#' Summarize a fitted symmetric epsilon-SVR with MAPE loss
#'
#' As [summary.psvr_mape()], with the symmetry parameter \eqn{a}{a} reported.
#'
#' @param object An object of class `"psvr_mape_sym"`, from [psvr_mape()] with
#'   `sym_type = "even"` or `"odd"`.
#' @param ... Ignored.
#'
#' @return `object`, invisibly. Called for the printed summary.
#'
#' @seealso [print.psvr_mape_sym()], [coef.psvr_mape_sym()]
#'
#' @examples
#' set.seed(1)
#' X <- matrix(rnorm(40), 20, 2)
#' y <- rlnorm(20)
#' fit <- psvr_mape(X, y, sym_type = "even",
#'                  kernel = make_kernel("rbf", sigma = 1), C = 10, eps = 5)
#' summary(fit)
#'
#' @export
summary.psvr_mape_sym <- function(object, ...)
  .summary_mape(object, "psvr_mape_sym")

#' Summarize a fitted LS-SVR with RMSPE loss
#'
#' Prints the kernel, the training count, the hyperparameter, and whether the
#' `diag(1/y)` preconditioner fired. No support-vector count is reported: LS-SVR
#' performs no pruning, so every training point contributes to the prediction.
#'
#' @param object An object of class `"psvr_rmspe"`, from [psvr_rmspe()] with
#'   `sym_type = "none"`, or from a parsnip fit unwrapped with
#'   [parsnip::extract_fit_engine()].
#' @param ... Ignored.
#'
#' @return `object`, invisibly. Called for the printed summary.
#'
#' @seealso [print.psvr_rmspe()], [coef.psvr_rmspe()]
#'
#' @examples
#' set.seed(1)
#' X <- matrix(rnorm(40), 20, 2)
#' y <- rlnorm(20)
#' fit <- psvr_rmspe(X, y, kernel = make_kernel("rbf", sigma = 1), gamma = 100)
#' summary(fit)
#'
#' @export
summary.psvr_rmspe <- function(object, ...) .summary_rmspe(object, "psvr_rmspe")

#' Summarize a fitted symmetric LS-SVR with RMSPE loss
#'
#' As [summary.psvr_rmspe()], with the symmetry parameter \eqn{a}{a} reported.
#'
#' @param object An object of class `"psvr_rmspe_sym"`, from [psvr_rmspe()]
#'   with `sym_type = "even"` or `"odd"`.
#' @param ... Ignored.
#'
#' @return `object`, invisibly. Called for the printed summary.
#'
#' @seealso [print.psvr_rmspe_sym()], [coef.psvr_rmspe_sym()]
#'
#' @examples
#' set.seed(1)
#' X <- matrix(rnorm(40), 20, 2)
#' y <- rlnorm(20)
#' fit <- psvr_rmspe(X, y, sym_type = "even",
#'                   kernel = make_kernel("rbf", sigma = 1), gamma = 100)
#' summary(fit)
#'
#' @export
summary.psvr_rmspe_sym <- function(object, ...)
  .summary_rmspe(object, "psvr_rmspe_sym")
