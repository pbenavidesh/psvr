# R/parsnip.R — parsnip integration for all four psvr models
#
# Parsnip type names use a `_model` suffix to avoid S3 dispatch ambiguity
# with the fitted object classes ("psvr_mape", etc.) that already have
# predict.* methods registered.
#
# User-facing constructors: psvr_mape(), psvr_mape_sym(),
#                           psvr_rmspe(), psvr_rmspe_sym()
# Each takes only the tunable hyperparameters; kernel (and `a` for symmetric
# models) are engine-specific and passed via set_engine("psvr", ...).

# ---- Fit wrappers --------------------------------------------------------
# parsnip calls the fit function with (x, y, ...) where x is a predictor
# matrix and y is the outcome vector.  Our underlying functions use (X, y).

#' @export
psvr_mape_fit <- function(x, y, kernel, C, eps, tol = 1e-5) {
  mape_svr(X = x, y = y, kernel = kernel, C = C, eps = eps, tol = tol)
}

#' @export
psvr_mape_sym_fit <- function(x, y, kernel, C, eps, a = 1L, tol = 1e-5) {
  mape_sym_svr(X = x, y = y, kernel = kernel, C = C, eps = eps,
               a = a, tol = tol)
}

#' @export
psvr_rmspe_fit <- function(x, y, kernel, gamma) {
  rmspe_lssvr(X = x, y = y, kernel = kernel, gamma = gamma)
}

#' @export
psvr_rmspe_sym_fit <- function(x, y, kernel, gamma, a = 1L) {
  rmspe_sym_lssvr(X = x, y = y, kernel = kernel, gamma = gamma, a = a)
}


# ---- Constructors --------------------------------------------------------

#' Parsnip model spec: epsilon-SVR with MAPE loss (Model 1)
#'
#' Creates a parsnip model specification for [mape_svr()].  The kernel and
#' solver tolerance are engine arguments passed via `set_engine()`.
#'
#' @param mode  Only `"regression"` is supported.
#' @param engine  Only `"psvr"` is available.
#' @param cost  Regularization parameter `C > 0`.  Use [tune()] to optimize.
#' @param svm_margin  Epsilon tube half-width `ε ≥ 0` (percentage units).
#'   Use [tune()] to optimize.
#'
#' @return A `psvr_mape_model` / `model_spec` object.
#'
#' @examples
#' \dontrun{
#' library(parsnip)
#' K <- make_kernel("rbf", sigma = 1)
#' spec <- psvr_mape(cost = 10, svm_margin = 5) |>
#'   set_engine("psvr", kernel = K)
#' fit(spec, mpg ~ ., data = mtcars)
#' }
#'
#' @export
psvr_mape <- function(mode = "regression", engine = "psvr",
                      cost = NULL, svm_margin = NULL) {
  args <- list(
    cost       = rlang::enquo(cost),
    svm_margin = rlang::enquo(svm_margin)
  )
  parsnip::new_model_spec(
    "psvr_mape_model",
    args                  = args,
    eng_args              = NULL,
    mode                  = mode,
    user_specified_mode   = !missing(mode),
    method                = NULL,
    engine                = engine,
    user_specified_engine = !missing(engine)
  )
}

#' Parsnip model spec: symmetric epsilon-SVR with MAPE loss (Model 2)
#'
#' Creates a parsnip model specification for [mape_sym_svr()].  The kernel,
#' symmetry parameter `a`, and solver tolerance are engine arguments.
#'
#' @param mode  Only `"regression"` is supported.
#' @param engine  Only `"psvr"` is available.
#' @param cost  Regularization parameter `C > 0`.
#' @param svm_margin  Epsilon tube half-width `ε ≥ 0` (percentage units).
#'
#' @return A `psvr_mape_sym_model` / `model_spec` object.
#'
#' @examples
#' \dontrun{
#' library(parsnip)
#' K <- make_kernel("rbf", sigma = 1)
#' spec <- psvr_mape_sym(cost = 10, svm_margin = 5) |>
#'   set_engine("psvr", kernel = K, a = 1L)
#' fit(spec, mpg ~ ., data = mtcars)
#' }
#'
#' @export
psvr_mape_sym <- function(mode = "regression", engine = "psvr",
                          cost = NULL, svm_margin = NULL) {
  args <- list(
    cost       = rlang::enquo(cost),
    svm_margin = rlang::enquo(svm_margin)
  )
  parsnip::new_model_spec(
    "psvr_mape_sym_model",
    args                  = args,
    eng_args              = NULL,
    mode                  = mode,
    user_specified_mode   = !missing(mode),
    method                = NULL,
    engine                = engine,
    user_specified_engine = !missing(engine)
  )
}

#' Parsnip model spec: LS-SVR with RMSPE loss (Model 3)
#'
#' Creates a parsnip model specification for [rmspe_lssvr()].  The kernel is
#' an engine argument passed via `set_engine()`.
#'
#' @param mode  Only `"regression"` is supported.
#' @param engine  Only `"psvr"` is available.
#' @param cost  Regularization parameter `Γ > 0`.  Use [tune()] to optimize.
#'
#' @return A `psvr_rmspe_model` / `model_spec` object.
#'
#' @examples
#' \dontrun{
#' library(parsnip)
#' K <- make_kernel("rbf", sigma = 1)
#' spec <- psvr_rmspe(cost = 1) |>
#'   set_engine("psvr", kernel = K)
#' fit(spec, mpg ~ ., data = mtcars)
#' }
#'
#' @export
psvr_rmspe <- function(mode = "regression", engine = "psvr",
                       cost = NULL) {
  args <- list(cost = rlang::enquo(cost))
  parsnip::new_model_spec(
    "psvr_rmspe_model",
    args                  = args,
    eng_args              = NULL,
    mode                  = mode,
    user_specified_mode   = !missing(mode),
    method                = NULL,
    engine                = engine,
    user_specified_engine = !missing(engine)
  )
}

#' Parsnip model spec: symmetric LS-SVR with RMSPE loss (Model 4)
#'
#' Creates a parsnip model specification for [rmspe_sym_lssvr()].  The kernel
#' and symmetry parameter `a` are engine arguments.
#'
#' @param mode  Only `"regression"` is supported.
#' @param engine  Only `"psvr"` is available.
#' @param cost  Regularization parameter `Γ > 0`.
#'
#' @return A `psvr_rmspe_sym_model` / `model_spec` object.
#'
#' @examples
#' \dontrun{
#' library(parsnip)
#' K <- make_kernel("rbf", sigma = 1)
#' spec <- psvr_rmspe_sym(cost = 1) |>
#'   set_engine("psvr", kernel = K, a = 1L)
#' fit(spec, mpg ~ ., data = mtcars)
#' }
#'
#' @export
psvr_rmspe_sym <- function(mode = "regression", engine = "psvr",
                            cost = NULL) {
  args <- list(cost = rlang::enquo(cost))
  parsnip::new_model_spec(
    "psvr_rmspe_sym_model",
    args                  = args,
    eng_args              = NULL,
    mode                  = mode,
    user_specified_mode   = !missing(mode),
    method                = NULL,
    engine                = engine,
    user_specified_engine = !missing(engine)
  )
}


# ---- Update methods -------------------------------------------------------

#' @export
update.psvr_mape_model <- function(object, parameters = NULL,
                                   cost = NULL, svm_margin = NULL,
                                   fresh = FALSE, ...) {
  args <- list(
    cost       = rlang::enquo(cost),
    svm_margin = rlang::enquo(svm_margin)
  )
  parsnip:::update_spec(object, parameters, args, fresh,
                        "psvr_mape_model", ...)
}

#' @export
update.psvr_mape_sym_model <- function(object, parameters = NULL,
                                       cost = NULL, svm_margin = NULL,
                                       fresh = FALSE, ...) {
  args <- list(
    cost       = rlang::enquo(cost),
    svm_margin = rlang::enquo(svm_margin)
  )
  parsnip:::update_spec(object, parameters, args, fresh,
                        "psvr_mape_sym_model", ...)
}

#' @export
update.psvr_rmspe_model <- function(object, parameters = NULL,
                                    cost = NULL,
                                    fresh = FALSE, ...) {
  args <- list(cost = rlang::enquo(cost))
  parsnip:::update_spec(object, parameters, args, fresh,
                        "psvr_rmspe_model", ...)
}

#' @export
update.psvr_rmspe_sym_model <- function(object, parameters = NULL,
                                        cost = NULL,
                                        fresh = FALSE, ...) {
  args <- list(cost = rlang::enquo(cost))
  parsnip:::update_spec(object, parameters, args, fresh,
                        "psvr_rmspe_sym_model", ...)
}


# ---- Engine registration -------------------------------------------------

make_psvr_engines <- function() {
  # Skip if already registered — parsnip's env persists across devtools reloads.
  if ("psvr_mape_model" %in% parsnip::get_from_env("models")) {
    return(invisible(NULL))
  }

  # ---- Model 1: psvr_mape_model / engine "psvr" ----

  parsnip::set_new_model("psvr_mape_model")
  parsnip::set_model_mode("psvr_mape_model", "regression")
  parsnip::set_model_engine("psvr_mape_model", mode = "regression",
                            eng = "psvr")
  parsnip::set_dependency("psvr_mape_model", eng = "psvr", pkg = "psvr")

  parsnip::set_model_arg(
    model        = "psvr_mape_model",
    eng          = "psvr",
    parsnip      = "cost",
    original     = "C",
    func         = list(pkg = "dials", fun = "cost"),
    has_submodel = FALSE
  )
  parsnip::set_model_arg(
    model        = "psvr_mape_model",
    eng          = "psvr",
    parsnip      = "svm_margin",
    original     = "eps",
    func         = list(pkg = "dials", fun = "svm_margin"),
    has_submodel = FALSE
  )

  parsnip::set_fit(
    model = "psvr_mape_model",
    eng   = "psvr",
    mode  = "regression",
    value = list(
      interface = "matrix",
      protect   = c("x", "y"),
      func      = c(pkg = "psvr", fun = "psvr_mape_fit"),
      defaults  = list()
    )
  )

  parsnip::set_encoding(
    model   = "psvr_mape_model",
    eng     = "psvr",
    mode    = "regression",
    options = list(
      predictor_indicators = "traditional",
      compute_intercept    = FALSE,
      remove_intercept     = FALSE,
      allow_sparse_x       = FALSE
    )
  )

  parsnip::set_pred(
    model = "psvr_mape_model",
    eng   = "psvr",
    mode  = "regression",
    type  = "numeric",
    value = list(
      pre  = NULL,
      post = NULL,
      func = c(fun = "predict"),
      args = list(
        object  = rlang::expr(object$fit),
        newdata = rlang::expr(new_data)
      )
    )
  )

  # ---- Model 2: psvr_mape_sym_model / engine "psvr" ----

  parsnip::set_new_model("psvr_mape_sym_model")
  parsnip::set_model_mode("psvr_mape_sym_model", "regression")
  parsnip::set_model_engine("psvr_mape_sym_model", mode = "regression",
                            eng = "psvr")
  parsnip::set_dependency("psvr_mape_sym_model", eng = "psvr", pkg = "psvr")

  parsnip::set_model_arg(
    model        = "psvr_mape_sym_model",
    eng          = "psvr",
    parsnip      = "cost",
    original     = "C",
    func         = list(pkg = "dials", fun = "cost"),
    has_submodel = FALSE
  )
  parsnip::set_model_arg(
    model        = "psvr_mape_sym_model",
    eng          = "psvr",
    parsnip      = "svm_margin",
    original     = "eps",
    func         = list(pkg = "dials", fun = "svm_margin"),
    has_submodel = FALSE
  )

  parsnip::set_fit(
    model = "psvr_mape_sym_model",
    eng   = "psvr",
    mode  = "regression",
    value = list(
      interface = "matrix",
      protect   = c("x", "y"),
      func      = c(pkg = "psvr", fun = "psvr_mape_sym_fit"),
      defaults  = list()
    )
  )

  parsnip::set_encoding(
    model   = "psvr_mape_sym_model",
    eng     = "psvr",
    mode    = "regression",
    options = list(
      predictor_indicators = "traditional",
      compute_intercept    = FALSE,
      remove_intercept     = FALSE,
      allow_sparse_x       = FALSE
    )
  )

  parsnip::set_pred(
    model = "psvr_mape_sym_model",
    eng   = "psvr",
    mode  = "regression",
    type  = "numeric",
    value = list(
      pre  = NULL,
      post = NULL,
      func = c(fun = "predict"),
      args = list(
        object  = rlang::expr(object$fit),
        newdata = rlang::expr(new_data)
      )
    )
  )

  # ---- Model 3: psvr_rmspe_model / engine "psvr" ----

  parsnip::set_new_model("psvr_rmspe_model")
  parsnip::set_model_mode("psvr_rmspe_model", "regression")
  parsnip::set_model_engine("psvr_rmspe_model", mode = "regression",
                            eng = "psvr")
  parsnip::set_dependency("psvr_rmspe_model", eng = "psvr", pkg = "psvr")

  parsnip::set_model_arg(
    model        = "psvr_rmspe_model",
    eng          = "psvr",
    parsnip      = "cost",
    original     = "gamma",
    func         = list(pkg = "dials", fun = "cost"),
    has_submodel = FALSE
  )

  parsnip::set_fit(
    model = "psvr_rmspe_model",
    eng   = "psvr",
    mode  = "regression",
    value = list(
      interface = "matrix",
      protect   = c("x", "y"),
      func      = c(pkg = "psvr", fun = "psvr_rmspe_fit"),
      defaults  = list()
    )
  )

  parsnip::set_encoding(
    model   = "psvr_rmspe_model",
    eng     = "psvr",
    mode    = "regression",
    options = list(
      predictor_indicators = "traditional",
      compute_intercept    = FALSE,
      remove_intercept     = FALSE,
      allow_sparse_x       = FALSE
    )
  )

  parsnip::set_pred(
    model = "psvr_rmspe_model",
    eng   = "psvr",
    mode  = "regression",
    type  = "numeric",
    value = list(
      pre  = NULL,
      post = NULL,
      func = c(fun = "predict"),
      args = list(
        object  = rlang::expr(object$fit),
        newdata = rlang::expr(new_data)
      )
    )
  )

  # ---- Model 4: psvr_rmspe_sym_model / engine "psvr" ----

  parsnip::set_new_model("psvr_rmspe_sym_model")
  parsnip::set_model_mode("psvr_rmspe_sym_model", "regression")
  parsnip::set_model_engine("psvr_rmspe_sym_model", mode = "regression",
                            eng = "psvr")
  parsnip::set_dependency("psvr_rmspe_sym_model", eng = "psvr", pkg = "psvr")

  parsnip::set_model_arg(
    model        = "psvr_rmspe_sym_model",
    eng          = "psvr",
    parsnip      = "cost",
    original     = "gamma",
    func         = list(pkg = "dials", fun = "cost"),
    has_submodel = FALSE
  )

  parsnip::set_fit(
    model = "psvr_rmspe_sym_model",
    eng   = "psvr",
    mode  = "regression",
    value = list(
      interface = "matrix",
      protect   = c("x", "y"),
      func      = c(pkg = "psvr", fun = "psvr_rmspe_sym_fit"),
      defaults  = list()
    )
  )

  parsnip::set_encoding(
    model   = "psvr_rmspe_sym_model",
    eng     = "psvr",
    mode    = "regression",
    options = list(
      predictor_indicators = "traditional",
      compute_intercept    = FALSE,
      remove_intercept     = FALSE,
      allow_sparse_x       = FALSE
    )
  )

  parsnip::set_pred(
    model = "psvr_rmspe_sym_model",
    eng   = "psvr",
    mode  = "regression",
    type  = "numeric",
    value = list(
      pre  = NULL,
      post = NULL,
      func = c(fun = "predict"),
      args = list(
        object  = rlang::expr(object$fit),
        newdata = rlang::expr(new_data)
      )
    )
  )
}


# ---- Package hook --------------------------------------------------------

.onLoad <- function(libname, pkgname) {
  if (requireNamespace("parsnip", quietly = TRUE)) {
    make_psvr_engines()
  }
}
