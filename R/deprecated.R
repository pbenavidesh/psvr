# Deprecated public API.  Each wrapper emits .Deprecated("psvr") and
# delegates to the matching internal `.fit_*` function (which still returns
# the legacy psvr_mape / psvr_mape_sym / psvr_rmspe / psvr_rmspe_sym
# shape with the legacy class). No shape conversion needed — the new
# psvr_fit shape lives only inside psvr() and its predict/print/coef/
# summary methods.
#
# Scheduled removal: v0.2.0 or later.

#' Fit epsilon-SVR with MAPE loss (deprecated)
#'
#' Soft-deprecated wrapper retained for backwards compatibility. New code
#' should use `psvr(loss = "mape", ...)`. Returns the legacy `psvr_mape`
#' object shape; the legacy `predict.psvr_mape()` / `print.psvr_mape()` /
#' `coef.psvr_mape()` methods continue to dispatch correctly.
#'
#' @param X,y Training matrix and strictly-positive target vector.
#' @param kernel Kernel closure from [make_kernel()].
#' @param C Regularization parameter `C > 0`.
#' @param eps Insensitivity tube half-width (% units), `eps >= 0`.
#' @param solver Backend: `"smo"` (default) or `"osqp"`.
#' @param tol Solver zero-threshold.
#'
#' @return A list of class `"psvr_mape"`.
#'
#' @examples
#' \dontrun{
#' # Use psvr() instead:
#' fit <- psvr(X, y, loss = "mape", kernel = make_kernel("rbf"),
#'             C = 10, eps = 5)
#' }
#'
#' @keywords internal
#' @export
mape_svr <- function(X, y, kernel, C, eps,
                     solver = c("smo", "osqp"), tol = 1e-5) {
  .Deprecated("psvr")
  .fit_mape(X = X, y = y, kernel = kernel, C = C, eps = eps,
            solver = match.arg(solver), tol = tol)
}

#' Fit symmetric epsilon-SVR with MAPE loss (deprecated)
#'
#' Soft-deprecated wrapper retained for backwards compatibility. New code
#' should use `psvr(loss = "mape", sym = +1L, ...)` (or `sym = -1L`).
#' Returns the legacy `psvr_mape_sym` object shape.
#'
#' @inheritParams mape_svr
#' @param a Symmetry parameter: `1` (even) or `-1` (odd).
#'
#' @return A list of class `"psvr_mape_sym"`.
#'
#' @examples
#' \dontrun{
#' fit <- psvr(X, y, loss = "mape", sym = +1L,
#'             kernel = make_kernel("rbf"), C = 10, eps = 5)
#' }
#'
#' @keywords internal
#' @export
mape_sym_svr <- function(X, y, kernel, C, eps, a = 1,
                         solver = c("smo", "osqp"), tol = 1e-5) {
  .Deprecated("psvr")
  .fit_mape_sym(X = X, y = y, kernel = kernel, C = C, eps = eps, a = a,
                solver = match.arg(solver), tol = tol)
}

#' Fit LS-SVR with RMSPE loss (deprecated)
#'
#' Soft-deprecated wrapper retained for backwards compatibility. New code
#' should use `psvr(loss = "rmspe", ...)`. Returns the legacy `psvr_rmspe`
#' object shape.
#'
#' @param X,y Training matrix and strictly-positive target vector.
#' @param kernel Kernel closure from [make_kernel()].
#' @param gamma Regularization parameter `Γ > 0`.
#' @param precondition One of `"auto"`, `"always"`, `"never"`, or a
#'   positive numeric threshold; controls Remark-17 symmetric rescaling.
#'
#' @return A list of class `"psvr_rmspe"`.
#'
#' @examples
#' \dontrun{
#' fit <- psvr(X, y, loss = "rmspe", kernel = make_kernel("rbf"),
#'             gamma = 100)
#' }
#'
#' @keywords internal
#' @export
rmspe_lssvr <- function(X, y, kernel, gamma, precondition = "auto") {
  .Deprecated("psvr")
  .fit_rmspe(X = X, y = y, kernel = kernel, gamma = gamma,
             precondition = precondition)
}

#' Fit symmetric LS-SVR with RMSPE loss (deprecated)
#'
#' Soft-deprecated wrapper retained for backwards compatibility. New code
#' should use `psvr(loss = "rmspe", sym = +1L, ...)` (or `sym = -1L`).
#' Returns the legacy `psvr_rmspe_sym` object shape.
#'
#' @inheritParams rmspe_lssvr
#' @param a Symmetry parameter: `1` (even) or `-1` (odd).
#'
#' @return A list of class `"psvr_rmspe_sym"`.
#'
#' @examples
#' \dontrun{
#' fit <- psvr(X, y, loss = "rmspe", sym = +1L,
#'             kernel = make_kernel("rbf"), gamma = 100)
#' }
#'
#' @keywords internal
#' @export
rmspe_sym_lssvr <- function(X, y, kernel, gamma, a = 1,
                            precondition = "auto") {
  .Deprecated("psvr")
  .fit_rmspe_sym(X = X, y = y, kernel = kernel, gamma = gamma, a = a,
                 precondition = precondition)
}
