# Internal validation helpers shared across the psvr fitters and the unified
# psvr() entry point.

# Validate strictly-positive targets. All percentage-error losses require y > 0.
.validate_y_positive <- function(y) {
  if (!all(y > 0)) {
    n_bad <- sum(y <= 0)
    stop(sprintf(
      paste0("%d target value%s non-positive (min = %g). ",
             "All targets must be strictly positive for percentage-error loss."),
      n_bad, if (n_bad == 1L) " is" else "s are", min(y)
    ))
  }
  invisible(NULL)
}

# Warn when N exceeds a threshold above which the dense kernel matrix becomes
# memory-heavy. The threshold default (2000) keeps the matrix under ~32 MB.
.warn_large_n <- function(N, threshold = 2000L) {
  if (N > threshold) {
    warning(sprintf(
      paste0("Large dataset (N = %d): kernel matrix is %d x %d (%.1f MB). ",
             "Consider subsampling for hyperparameter tuning."),
      N, N, N, N^2 * 8 / 1e6
    ))
  }
  invisible(NULL)
}

# Validate inputs for the unified psvr() entry point. Used in Step 4+.
#
# `passed` is a named logical vector of `missing()` flags from psvr()'s frame:
# TRUE means the user supplied that argument (so a cross-loss mismatch should
# warn). `match.arg` defaults are indistinguishable from user input by value
# alone, so the call site must compute these flags before delegating here.
.validate_psvr_inputs <- function(X, y, loss, sym,
                                  C = NULL, eps = NULL,
                                  gamma = NULL,
                                  a = NULL,
                                  alpha_init = NULL,
                                  alpha_star_init = NULL,
                                  reg = NULL,
                                  passed = list()) {

  if (!is.null(reg))
    stop("psvr 0.0.2.9004 does not implement extended Lagrangian (`reg`); ",
         "planned for a future phase. Pass `reg = NULL`.")

  has_warm <- !is.null(alpha_init) || !is.null(alpha_star_init)
  if (has_warm && loss == "rmspe") {
    stop("Warm-start is not supported for `loss = \"rmspe\"` (LS-SVR is a ",
         "single linear-system solve; there is no SMO state to carry over). ",
         "Use `loss = \"mape\"` for warm-start, or `tune::tune_grid()` with ",
         "parallel cold-start for RMSPE cross-validation.")
  }
  if (has_warm) {
    N <- nrow(X)
    if (!is.null(alpha_init)) {
      if (!is.numeric(alpha_init) || length(alpha_init) != N ||
          any(!is.finite(alpha_init)))
        stop("`alpha_init` must be a finite numeric vector of length nrow(X).")
    }
    if (!is.null(alpha_star_init)) {
      if (!is.numeric(alpha_star_init) || length(alpha_star_init) != N ||
          any(!is.finite(alpha_star_init)))
        stop("`alpha_star_init` must be a finite numeric vector of length nrow(X).")
    }
  }

  if (!loss %in% c("mape", "rmspe"))
    stop('`loss` must be one of "mape" or "rmspe"')

  if (!is.null(sym)) {
    sym_int <- suppressWarnings(as.integer(sym))
    if (length(sym_int) != 1L || is.na(sym_int) || !sym_int %in% c(-1L, 1L))
      stop("`sym` must be NULL, +1L, or -1L")
  }

  if (!is.null(a) && (length(a) != 1L || !a %in% c(-1L, 1L)))
    stop("`a` must be 1 (even) or -1 (odd)")

  .validate_y_positive(y)

  if (loss == "mape") {
    if (is.null(C))   stop('`C` is required when `loss = "mape"`')
    if (is.null(eps)) stop('`eps` is required when `loss = "mape"`')
    if (C   <= 0) stop("`C` must be positive")
    if (eps <  0) stop("`eps` must be non-negative")
    # Cross-loss: warn only when the user actively supplied a non-NULL value
    # for an LS-SVR-only arg.  (Explicit `gamma = NULL` ≡ not passing.)
    if (isTRUE(passed$gamma) && !is.null(gamma))
      warning('`gamma` is ignored when `loss = "mape"`')
    if (isTRUE(passed$precondition))
      warning('`precondition` is ignored when `loss = "mape"`')
  } else {  # rmspe
    if (is.null(gamma)) stop('`gamma` is required when `loss = "rmspe"`')
    if (gamma <= 0)     stop("`gamma` must be positive")
    if (isTRUE(passed$C)   && !is.null(C))   warning('`C` is ignored when `loss = "rmspe"`')
    if (isTRUE(passed$eps) && !is.null(eps)) warning('`eps` is ignored when `loss = "rmspe"`')
    if (isTRUE(passed$solver))               warning('`solver` is ignored when `loss = "rmspe"`')
    if (isTRUE(passed$tol))                  warning('`tol` is ignored when `loss = "rmspe"`')
  }

  invisible(NULL)
}
