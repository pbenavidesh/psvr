# fitted() and residuals() for the four fit classes.
#
# The shared Rd topics are named `psvr-fitted` / `psvr-residuals` rather than
# hosted on one method: before API-redesign stage 5 they were hosted on
# fitted.psvr_fit / residuals.psvr_fit, and deleting that class would otherwise
# have promoted an arbitrary one of the four survivors to topic owner.
#
# Both read the length-N `y_train` / `fitted_values` vectors stored at fit
# time; neither recomputes predictions, so neither rebuilds the N x N kernel
# matrix. See .fit_mape() / .fit_rmspe() for how the fitted values are
# recovered from solver state.
#
# ASCII-only by design: this file is checked by R CMD check's
# .check_package_ASCII_code(), which permits non-ASCII in comments but not in
# code (the near-zero warning below is a string literal). Mathematical
# notation in the roxygen below uses Rd \eqn{} markup rather than Unicode or
# \uxxxx escapes, so it renders on every platform.

# Stale-object guard. Objects fitted before psvr 0.0.2.9010 have neither
# field; without this they would fail inside arithmetic on NULL with a
# message that says nothing about the cause.
.psvr_require_training_state <- function(object, what) {
  if (is.null(object$fitted_values) || is.null(object$y_train)) {
    stop(sprintf(
      paste0("`%s()` needs the training targets and fitted values, which this ",
             "%s object\ndoes not carry: it was fitted with psvr < 0.0.2.9010, ",
             "before those fields were added.\nRefit the model with the current ",
             "version to use %s(), e.g. `fit <- psvr_mape(X, y, ...)` or ",
             "`psvr_rmspe(X, y, ...)`."),
      what, class(object)[1L], what),
      call. = FALSE)
  }
  invisible(NULL)
}

.psvr_fitted <- function(object) {
  .psvr_require_training_state(object, "fitted")
  as.numeric(object$fitted_values)
}

# Relative floor below which a fitted value makes (y - yhat)/yhat
# uninformative. Scaled by mean(y) so it tracks the units of the response.
.psvr_yhat_floor <- function(y) sqrt(.Machine$double.eps) * mean(abs(y))

.psvr_residuals <- function(object, type) {
  .psvr_require_training_state(object, "residuals")
  type <- match.arg(type, c("response", "percentage", "multiplicative"))

  y    <- as.numeric(object$y_train)
  yhat <- as.numeric(object$fitted_values)
  r    <- y - yhat

  switch(
    type,
    response   = r,
    # Divides by the OBSERVED target: this is the per-observation term of
    # the MAPE loss the model was fitted under.
    percentage = r / y,
    # Divides by the FITTED value: the multiplicative-noise eta-hat.
    multiplicative = {
      bad <- !is.finite(yhat) | yhat <= .psvr_yhat_floor(y)
      if (any(bad)) {
        warning(sprintf(
          paste0("%d of %d fitted value%s at or below the near-zero floor ",
                 "(%.3g); the\n  corresponding \"multiplicative\" residuals are ",
                 "inflated, infinite, or sign-flipped.\n  Returned unmodified; ",
                 "see ?psvr-residuals for the exclusion rule used for ",
                 "pooled diagnostics."),
          sum(bad), length(bad), if (sum(bad) == 1L) " is" else "s are",
          .psvr_yhat_floor(y)),
          call. = FALSE)
      }
      r / yhat
    }
  )
}

#' Extract training fitted values from a psvr model
#'
#' Returns the length-`N` in-sample predictions \eqn{f(x_k)}{f(x_k)} recorded
#' when the model was fitted. No kernel matrix is rebuilt: the values are
#' recovered from state the solver already holds. For the MAPE models that is
#' a matvec against the retained \eqn{\Omega}{Omega}; for the LS-SVR models it
#' is the KKT stationarity identity
#' \deqn{f(x_k) = y_k - (10^{-6} + y_k^2/\Gamma)\,\alpha_k}{%
#'       f(x_k) = y_k - (1e-6 + y_k^2/Gamma) * alpha_k}
#' which costs \eqn{O(N)}{O(N)} and holds in both preconditioner branches.
#' The training inputs `X` are not retained for this purpose.
#'
#' The result equals `predict(object, X_train)` to machine precision. It is
#' not bit-identical: the two use different summation orders (a BLAS matvec
#' versus the column-wise reduction in `predict()`), and for the LS-SVR
#' models the identity above is exact only up to the residual of the linear
#' solve. Observed agreement is within `3e-12` relative across the four
#' models.
#'
#' @section Not reachable through parsnip:
#' parsnip registers neither `residuals.model_fit` nor `fitted.model_fit`, so
#' calling either generic on a `model_fit` dispatches to the stats default and
#' returns `NULL` **silently** - no error, no warning. Reach the psvr object
#' first with [parsnip::extract_fit_engine()], then call `fitted()` or
#' `residuals()` on that. psvr deliberately does not register S3 methods on
#' parsnip's class. Note also that `parsnip::augment()` recomputes predictions
#' on whatever `new_data` it is given and reports response residuals only.
#'
#' @param object A fitted object of class `"psvr_mape"`, `"psvr_mape_sym"`,
#'   `"psvr_rmspe"` or `"psvr_rmspe_sym"`, from [psvr_mape()], [psvr_rmspe()],
#'   or a parsnip fit unwrapped with [parsnip::extract_fit_engine()].
#' @param ... Ignored.
#'
#' @return Numeric vector of length `N` (the number of training
#'   observations), in training-row order.
#'
#' @seealso [residuals.psvr_mape()] and the other residuals methods
#'
#' @examples
#' set.seed(1)
#' X <- matrix(runif(40, 0.5, 3), 20, 2)
#' y <- 2 + X[, 1]^2
#' fit <- psvr_rmspe(X, y, kernel = make_kernel("rbf"), gamma = 100)
#' head(fitted(fit))
#'
#' # Through parsnip both generics return NULL on the model_fit wrapper;
#' # extract the engine object first.
#' df   <- data.frame(x1 = X[, 1], x2 = X[, 2], y = y)
#' spec <- psvr_rmspe_rbf(cost = 10, rbf_sigma = 0.8)
#' pfit <- parsnip::fit(spec, y ~ x1 + x2, data = df)
#' fitted(pfit)                                     # NULL
#' head(fitted(parsnip::extract_fit_engine(pfit)))  # the fitted values
#'
#' @importFrom stats fitted residuals
#' @name psvr-fitted
NULL

#' Extract training residuals from a psvr model
#'
#' Three residual types are available. They are **different quantities with
#' different denominators**, not interchangeable scalings of one another:
#'
#' \describe{
#'   \item{`"response"`}{\eqn{y - \hat{y}}{y - yhat}. The default, following
#'     the R convention for [stats::residuals()]. Units of the response.}
#'   \item{`"percentage"`}{\eqn{(y - \hat{y}) / y}{(y - yhat) / y}. Divides by
#'     the **observed target**. This is the per-observation contribution to
#'     the MAPE loss the epsilon-SVR models are fitted under, so it is the
#'     residual that corresponds to the estimated objective.
#'     `mean(abs(.)) * 100` is the training MAPE.}
#'   \item{`"multiplicative"`}{\eqn{(y - \hat{y}) / \hat{y}}{(y - yhat) /
#'     yhat}. Divides by the **fitted value**. This is the
#'     \eqn{\hat{\eta}}{eta-hat} of the multiplicative-noise model
#'     \eqn{Y = f(x)(1 + \eta)}{Y = f(x)(1 + eta)}, under which
#'     \eqn{\eta = (Y - f(x))/f(x)}{eta = (Y - f(x))/f(x)}. Use it to inspect
#'     the assumed noise structure, e.g. checking whether
#'     \eqn{\hat{\eta}}{eta-hat} is homoscedastic and centred at zero.}
#' }
#'
#' The denominators differ and the choice matters: `"percentage"` divides by
#' the observed target because that is what the MAPE loss does,
#' `"multiplicative"` divides by the fitted value because that is what the
#' noise model does. They agree only when \eqn{y = \hat{y}}{y = yhat}, and
#' diverge as the fit degrades. Choose by the question being asked: the loss
#' actually minimised (`"percentage"`), or the noise model assumed
#' (`"multiplicative"`).
#'
#' @section Near-zero fitted values:
#' `"multiplicative"` divides by \eqn{\hat{y}}{yhat}, which an SVR does not
#' constrain to be positive even though the targets are. Where
#' \eqn{\hat{y}}{yhat} is at or below `sqrt(.Machine$double.eps) *
#' mean(abs(y))` - including zero and negative fitted values - the ratio is
#' inflated, infinite, or sign-flipped.
#'
#' This function does **not** drop those observations: it always returns
#' exactly `N` values in training-row order, so the result stays aligned
#' with `y_train`, `fitted()`, and the training rows. The raw value
#' (possibly `Inf` or `NaN`) is returned and a single warning reports how
#' many observations are affected.
#'
#' For *pooled* diagnostics (a mean multiplicative error, a variance, a
#' histogram) the affected observations must be excluded, or one near-zero
#' fitted value dominates the summary. This follows the standard treatment of
#' percentage errors near zero (Makridakis); the thesis applies it in Ch. 7
#' Sec. 7.6.1 by excluding observations below a threshold before pooling.
#' Exclude at the point of aggregation, e.g.:
#'
#' ```
#' e <- residuals(fit, type = "multiplicative")
#' mean(e[is.finite(e)])
#' ```
#'
#' @section Not reachable through parsnip:
#' parsnip registers neither `residuals.model_fit` nor `fitted.model_fit`, so
#' calling either generic on a `model_fit` dispatches to the stats default and
#' returns `NULL` **silently** - no error, no warning. Reach the psvr object
#' first with [parsnip::extract_fit_engine()], then call `residuals()` or
#' `fitted()` on that. psvr deliberately does not register S3 methods on
#' parsnip's class. Note also that `parsnip::augment()` recomputes predictions
#' on whatever `new_data` it is given and reports response residuals only, so
#' it is not a substitute for the `"percentage"` and `"multiplicative"` types.
#'
#' @param object A fitted object of class `"psvr_mape"`, `"psvr_mape_sym"`,
#'   `"psvr_rmspe"` or `"psvr_rmspe_sym"`, from [psvr_mape()], [psvr_rmspe()],
#'   or a parsnip fit unwrapped with [parsnip::extract_fit_engine()].
#' @param type One of `"response"` (default), `"percentage"`, or
#'   `"multiplicative"`. See above; the denominators differ.
#' @param ... Ignored.
#'
#' @return Numeric vector of length `N` (the number of training
#'   observations), in training-row order.
#'
#' @seealso [fitted.psvr_mape()] and the other fitted methods
#'
#' @examples
#' set.seed(1)
#' X <- matrix(runif(40, 0.5, 3), 20, 2)
#' y <- 2 + X[, 1]^2
#' fit <- psvr_rmspe(X, y, kernel = make_kernel("rbf"), gamma = 100)
#' head(residuals(fit))
#' mean(abs(residuals(fit, type = "percentage"))) * 100   # training MAPE
#'
#' # Through parsnip both generics return NULL on the model_fit wrapper;
#' # extract the engine object first.
#' df   <- data.frame(x1 = X[, 1], x2 = X[, 2], y = y)
#' spec <- psvr_rmspe_rbf(cost = 10, rbf_sigma = 0.8)
#' pfit <- parsnip::fit(spec, y ~ x1 + x2, data = df)
#' residuals(pfit)                                     # NULL
#' head(residuals(parsnip::extract_fit_engine(pfit)))  # the residuals
#'
#' @name psvr-residuals
NULL

#' @rdname psvr-fitted
#' @export
fitted.psvr_mape <- function(object, ...) .psvr_fitted(object)

#' @rdname psvr-fitted
#' @export
fitted.psvr_mape_sym <- function(object, ...) .psvr_fitted(object)

#' @rdname psvr-fitted
#' @export
fitted.psvr_rmspe <- function(object, ...) .psvr_fitted(object)

#' @rdname psvr-fitted
#' @export
fitted.psvr_rmspe_sym <- function(object, ...) .psvr_fitted(object)

#' @rdname psvr-residuals
#' @export
residuals.psvr_mape <- function(object,
                                type = c("response", "percentage",
                                         "multiplicative"), ...) {
  .psvr_residuals(object, type)
}

#' @rdname psvr-residuals
#' @export
residuals.psvr_mape_sym <- function(object,
                                    type = c("response", "percentage",
                                             "multiplicative"), ...) {
  .psvr_residuals(object, type)
}

#' @rdname psvr-residuals
#' @export
residuals.psvr_rmspe <- function(object,
                                 type = c("response", "percentage",
                                          "multiplicative"), ...) {
  .psvr_residuals(object, type)
}

#' @rdname psvr-residuals
#' @export
residuals.psvr_rmspe_sym <- function(object,
                                     type = c("response", "percentage",
                                              "multiplicative"), ...) {
  .psvr_residuals(object, type)
}
