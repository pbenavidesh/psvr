# Internal validation helpers shared across the psvr fitters and the two
# public entry points, psvr_mape() and psvr_rmspe().

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

# Validate the warm-start vectors for psvr_mape().
#
# This is the whole of what the epsilon-SVR entry point adds over its fitters:
# .fit_mape() / .fit_mape_sym() already check y > 0, C > 0, eps >= 0 and
# a in {-1, 1} themselves, and duplicating those here would be drift.
#
# There is deliberately NO .validate_rmspe_inputs() sibling. The LS-SVR entry
# point adds nothing its fitters do not already check except the presence of
# `gamma` and `kernel` -- and presence is tested with missing(), which only
# works in the frame that owns the formal and therefore cannot be delegated to
# a helper at all. A one-line rmspe validator would have been a wrapper around
# nothing.
#
# Message wording is load-bearing: test-warm-start.R matches on the substrings
# "finite numeric" and "length nrow(X)".
.validate_mape_inputs <- function(X, alpha_init = NULL, alpha_star_init = NULL) {
  if (is.null(alpha_init) && is.null(alpha_star_init)) return(invisible(NULL))

  N <- nrow(X)
  chk <- function(v, nm) {
    if (is.null(v)) return(invisible(NULL))
    if (!is.numeric(v) || length(v) != N || any(!is.finite(v)))
      stop(sprintf("`%s` must be a finite numeric vector of length nrow(X).", nm),
           call. = FALSE)
    invisible(NULL)
  }
  chk(alpha_init,      "alpha_init")
  chk(alpha_star_init, "alpha_star_init")
  invisible(NULL)
}

