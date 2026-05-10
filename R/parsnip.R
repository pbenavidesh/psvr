# R/parsnip.R — parsnip integration for psvr: 12 model specs
#
# Naming convention: psvr_{loss}_{kernel}()  — 3 kernels × 4 models
#   loss   ∈ {mape, mape_sym, rmspe, rmspe_sym}
#   kernel ∈ {rbf, poly, linear}
#
# Kernel parameters are tunable parsnip args mapped to standard dials params:
#   RBF:  rbf_sigma → dials::rbf_sigma()
#   Poly: degree    → dials::degree(),  scale_factor → dials::scale_factor()
#   Linear: (none)
#
# The symmetry parameter for symmetric models is exposed as the tunable
# model argument `sym_type` ("even" → a = 1L, "odd" → a = -1L).  Fit
# wrappers translate it to integer a before calling the underlying solver.

utils::globalVariables(c("object", "new_data"))


# ---- Fit wrappers --------------------------------------------------------
# parsnip calls each wrapper with (x, y, <original-arg-names>, ...).
# The wrapper builds the kernel and delegates to the underlying internal
# fitter. The wrappers must be EXPORTED (parsnip's set_fit resolves
# `c(pkg, fun)` via `pkg::fun`, which only sees exported objects), but
# they are tagged `@keywords internal` so they are hidden from the
# pkgdown reference index and not advertised as user API.

#' @title Fit wrappers for parsnip engine dispatch
#' @description
#' Bridge functions called by parsnip when fitting psvr model specs.
#' Exported only because parsnip's resolver requires it; not intended
#' for direct use. Call [psvr()] instead for direct fitting.
#' @param x Numeric predictor matrix (parsnip matrix interface).
#' @param y Numeric outcome vector (strictly positive).
#' @param C Regularization parameter for MAPE models.
#' @param eps Epsilon tube half-width for MAPE models.
#' @param gamma Regularization parameter for RMSPE models.
#' @param rbf_sigma RBF bandwidth σ > 0.
#' @param degree Polynomial degree ≥ 1.
#' @param scale_factor Polynomial constant term (coef₀).
#' @param sym_type Symmetry type (`"even"` or `"odd"`) for symmetric models;
#'   translated to `a = 1L` or `a = -1L` before calling the solver.
#' @param tol Solver zero-threshold.
#' @param precondition Optional symmetric rescaling preconditioner for the
#'   RMSPE LS-SVR fitters. See [rmspe_lssvr()] for accepted values and
#'   semantics.
#' @name psvr-fit-wrappers
#' @keywords internal
NULL

#' @rdname psvr-fit-wrappers
#' @keywords internal
#' @export
psvr_mape_rbf_fit <- function(x, y, C, eps, rbf_sigma = 1, tol = 1e-5) {
  .fit_mape(X = x, y = y,
            kernel = make_kernel("rbf", sigma = rbf_sigma),
            C = C, eps = eps, tol = tol)
}

#' @rdname psvr-fit-wrappers
#' @keywords internal
#' @export
psvr_mape_poly_fit <- function(x, y, C, eps, degree = 3L, scale_factor = 1,
                               tol = 1e-5) {
  .fit_mape(X = x, y = y,
            kernel = make_kernel("polynomial", degree = degree,
                                 coef0 = scale_factor),
            C = C, eps = eps, tol = tol)
}

#' @rdname psvr-fit-wrappers
#' @keywords internal
#' @export
psvr_mape_linear_fit <- function(x, y, C, eps, tol = 1e-5) {
  .fit_mape(X = x, y = y,
            kernel = make_kernel("linear"),
            C = C, eps = eps, tol = tol)
}

#' @rdname psvr-fit-wrappers
#' @keywords internal
#' @export
psvr_mape_sym_rbf_fit <- function(x, y, C, eps, rbf_sigma = 1,
                                  sym_type = "even", tol = 1e-5) {
  a <- if (sym_type == "even") 1L else -1L
  .fit_mape_sym(X = x, y = y,
                kernel = make_kernel("rbf", sigma = rbf_sigma),
                C = C, eps = eps, a = a, tol = tol)
}

#' @rdname psvr-fit-wrappers
#' @keywords internal
#' @export
psvr_mape_sym_poly_fit <- function(x, y, C, eps, degree = 3L,
                                   scale_factor = 1, a = 1L, tol = 1e-5) {
  .fit_mape_sym(X = x, y = y,
                kernel = make_kernel("polynomial", degree = degree,
                                     coef0 = scale_factor),
                C = C, eps = eps, a = a, tol = tol)
}

#' @rdname psvr-fit-wrappers
#' @keywords internal
#' @export
psvr_mape_sym_linear_fit <- function(x, y, C, eps, a = 1L, tol = 1e-5) {
  .fit_mape_sym(X = x, y = y,
                kernel = make_kernel("linear"),
                C = C, eps = eps, a = a, tol = tol)
}

#' @rdname psvr-fit-wrappers
#' @keywords internal
#' @export
psvr_rmspe_rbf_fit <- function(x, y, gamma, rbf_sigma = 1,
                               precondition = "auto") {
  .fit_rmspe(X = x, y = y,
             kernel = make_kernel("rbf", sigma = rbf_sigma),
             gamma = gamma, precondition = precondition)
}

#' @rdname psvr-fit-wrappers
#' @keywords internal
#' @export
psvr_rmspe_poly_fit <- function(x, y, gamma, degree = 3L, scale_factor = 1,
                                precondition = "auto") {
  .fit_rmspe(X = x, y = y,
             kernel = make_kernel("polynomial", degree = degree,
                                  coef0 = scale_factor),
             gamma = gamma, precondition = precondition)
}

#' @rdname psvr-fit-wrappers
#' @keywords internal
#' @export
psvr_rmspe_linear_fit <- function(x, y, gamma, precondition = "auto") {
  .fit_rmspe(X = x, y = y, kernel = make_kernel("linear"),
             gamma = gamma, precondition = precondition)
}

#' @rdname psvr-fit-wrappers
#' @keywords internal
#' @export
psvr_rmspe_sym_rbf_fit <- function(x, y, gamma, rbf_sigma = 1,
                                   sym_type = "even",
                                   precondition = "auto") {
  a <- if (sym_type == "even") 1L else -1L
  .fit_rmspe_sym(X = x, y = y,
                 kernel = make_kernel("rbf", sigma = rbf_sigma),
                 gamma = gamma, a = a, precondition = precondition)
}

#' @rdname psvr-fit-wrappers
#' @keywords internal
#' @export
psvr_rmspe_sym_poly_fit <- function(x, y, gamma, degree = 3L,
                                    scale_factor = 1, a = 1L,
                                    precondition = "auto") {
  .fit_rmspe_sym(X = x, y = y,
                 kernel = make_kernel("polynomial", degree = degree,
                                      coef0 = scale_factor),
                 gamma = gamma, a = a, precondition = precondition)
}

#' @rdname psvr-fit-wrappers
#' @keywords internal
#' @export
psvr_rmspe_sym_linear_fit <- function(x, y, gamma, a = 1L,
                                      precondition = "auto") {
  .fit_rmspe_sym(X = x, y = y, kernel = make_kernel("linear"),
                 gamma = gamma, a = a, precondition = precondition)
}


# ---- Constructors --------------------------------------------------------

#' Parsnip model specs: epsilon-SVR with MAPE loss (Model 1)
#'
#' Create parsnip model specifications for [mape_svr()] with a fixed kernel
#' type.  Kernel parameters are tunable parsnip arguments; the symmetry
#' parameter `a` and solver tolerance are engine arguments passed via
#' `set_engine()`.
#'
#' @param mode   Only `"regression"` is supported.
#' @param engine Only `"psvr"` is available.
#' @param cost   Regularization parameter `C > 0`.  Use [tune()] to optimize.
#'   Mapped to [cost_psvr()] with range `[-2, 10]` on the log2 scale — wider
#'   than `dials::cost()` to cover the larger values needed by LS-SVR models.
#' @param svm_margin Epsilon tube half-width `ε ≥ 0` expressed as a percentage
#'   of each target value.  Use [tune()] to optimize.  Mapped to
#'   [margin_percentage()] with default range `[1, 20]` (percentage units).
#' @param rbf_sigma RBF bandwidth σ > 0.  Use [tune()] to optimize.
#'   Mapped to [rbf_sigma_psvr()]; the search range auto-finalizes using the
#'   median-distance heuristic when training data are available.
#'   (RBF specs only.)
#' @param degree Polynomial degree ≥ 1.  Use [tune()] to optimize.
#'   (Polynomial specs only.)
#' @param scale_factor Polynomial constant term (coef₀).  Use [tune()] to
#'   optimize.  (Polynomial specs only.)
#'
#' @return A parsnip `model_spec` object of the corresponding class.
#'
#' @examples
#' \dontrun{
#' library(parsnip)
#' spec <- psvr_mape_rbf(cost = 10, svm_margin = 1, rbf_sigma = 1) |>
#'   set_engine("psvr")
#'
#' spec_poly <- psvr_mape_poly(cost = 10, svm_margin = 1, degree = 2,
#'                             scale_factor = 1) |>
#'   set_engine("psvr")
#'
#' spec_lin <- psvr_mape_linear(cost = 10, svm_margin = 1) |>
#'   set_engine("psvr")
#' }
#'
#' @name psvr_mape_specs
#' @export
psvr_mape_rbf <- function(mode = "regression", engine = "psvr",
                          cost = NULL, svm_margin = NULL, rbf_sigma = NULL) {
  args <- list(
    cost       = rlang::enquo(cost),
    svm_margin = rlang::enquo(svm_margin),
    rbf_sigma  = rlang::enquo(rbf_sigma)
  )
  parsnip::new_model_spec(
    "psvr_mape_rbf_model",
    args                  = args,
    eng_args              = NULL,
    mode                  = mode,
    user_specified_mode   = !missing(mode),
    method                = NULL,
    engine                = engine,
    user_specified_engine = !missing(engine)
  )
}

#' @rdname psvr_mape_specs
#' @export
psvr_mape_poly <- function(mode = "regression", engine = "psvr",
                           cost = NULL, svm_margin = NULL,
                           degree = NULL, scale_factor = NULL) {
  args <- list(
    cost         = rlang::enquo(cost),
    svm_margin   = rlang::enquo(svm_margin),
    degree       = rlang::enquo(degree),
    scale_factor = rlang::enquo(scale_factor)
  )
  parsnip::new_model_spec(
    "psvr_mape_poly_model",
    args                  = args,
    eng_args              = NULL,
    mode                  = mode,
    user_specified_mode   = !missing(mode),
    method                = NULL,
    engine                = engine,
    user_specified_engine = !missing(engine)
  )
}

#' @rdname psvr_mape_specs
#' @export
psvr_mape_linear <- function(mode = "regression", engine = "psvr",
                             cost = NULL, svm_margin = NULL) {
  args <- list(
    cost       = rlang::enquo(cost),
    svm_margin = rlang::enquo(svm_margin)
  )
  parsnip::new_model_spec(
    "psvr_mape_linear_model",
    args                  = args,
    eng_args              = NULL,
    mode                  = mode,
    user_specified_mode   = !missing(mode),
    method                = NULL,
    engine                = engine,
    user_specified_engine = !missing(engine)
  )
}

#' Parsnip model specs: symmetric epsilon-SVR with MAPE loss (Model 2)
#'
#' Create parsnip model specifications for [mape_sym_svr()] with a fixed
#' kernel type.  The symmetry type is exposed as the tunable `sym_type`
#' argument (`"even"` for a = 1, `"odd"` for a = -1); pass
#' `sym_type = tune()` to let CV select it automatically.
#'
#' @inheritParams psvr_mape_specs
#' @param sym_type Symmetry type: `"even"` (default, a = 1) or `"odd"`
#'   (a = -1).  Use [tune()] to optimise over both values during CV.
#'
#' @return A parsnip `model_spec` object of the corresponding class.
#'
#' @examples
#' \dontrun{
#' library(parsnip)
#' spec <- psvr_mape_sym_rbf(cost = 10, svm_margin = 1, rbf_sigma = 1) |>
#'   set_engine("psvr")
#'
#' spec_poly <- psvr_mape_sym_poly(cost = 10, svm_margin = 1, degree = 2,
#'                                 scale_factor = 1) |>
#'   set_engine("psvr")
#'
#' spec_lin <- psvr_mape_sym_linear(cost = 10, svm_margin = 1) |>
#'   set_engine("psvr")
#' }
#'
#' @name psvr_mape_sym_specs
#' @export
psvr_mape_sym_rbf <- function(mode = "regression", engine = "psvr",
                              cost = NULL, svm_margin = NULL,
                              rbf_sigma = NULL, sym_type = NULL) {
  args <- list(
    cost       = rlang::enquo(cost),
    svm_margin = rlang::enquo(svm_margin),
    rbf_sigma  = rlang::enquo(rbf_sigma),
    sym_type   = rlang::enquo(sym_type)
  )
  parsnip::new_model_spec(
    "psvr_mape_sym_rbf_model",
    args                  = args,
    eng_args              = NULL,
    mode                  = mode,
    user_specified_mode   = !missing(mode),
    method                = NULL,
    engine                = engine,
    user_specified_engine = !missing(engine)
  )
}

#' @rdname psvr_mape_sym_specs
#' @export
psvr_mape_sym_poly <- function(mode = "regression", engine = "psvr",
                               cost = NULL, svm_margin = NULL,
                               degree = NULL, scale_factor = NULL) {
  args <- list(
    cost         = rlang::enquo(cost),
    svm_margin   = rlang::enquo(svm_margin),
    degree       = rlang::enquo(degree),
    scale_factor = rlang::enquo(scale_factor)
  )
  parsnip::new_model_spec(
    "psvr_mape_sym_poly_model",
    args                  = args,
    eng_args              = NULL,
    mode                  = mode,
    user_specified_mode   = !missing(mode),
    method                = NULL,
    engine                = engine,
    user_specified_engine = !missing(engine)
  )
}

#' @rdname psvr_mape_sym_specs
#' @export
psvr_mape_sym_linear <- function(mode = "regression", engine = "psvr",
                                 cost = NULL, svm_margin = NULL) {
  args <- list(
    cost       = rlang::enquo(cost),
    svm_margin = rlang::enquo(svm_margin)
  )
  parsnip::new_model_spec(
    "psvr_mape_sym_linear_model",
    args                  = args,
    eng_args              = NULL,
    mode                  = mode,
    user_specified_mode   = !missing(mode),
    method                = NULL,
    engine                = engine,
    user_specified_engine = !missing(engine)
  )
}

#' Parsnip model specs: LS-SVR with RMSPE loss (Model 3)
#'
#' Create parsnip model specifications for [rmspe_lssvr()] with a fixed kernel
#' type.  `cost` maps to the regularization parameter `Γ`.
#'
#' @param mode   Only `"regression"` is supported.
#' @param engine Only `"psvr"` is available.
#' @param cost   Regularization parameter `Γ > 0`.  Use [tune()] to optimize.
#'   Mapped to [cost_psvr()] with range `[-2, 10]` on the log2 scale — wider
#'   than `dials::cost()` to cover the larger values needed by LS-SVR models.
#' @param rbf_sigma RBF bandwidth σ > 0.  Use [tune()] to optimize.
#'   Mapped to [rbf_sigma_psvr()]; the search range auto-finalizes using the
#'   median-distance heuristic when training data are available.
#'   (RBF specs only.)
#' @param degree Polynomial degree ≥ 1.  Use [tune()] to optimize.
#'   (Polynomial specs only.)
#' @param scale_factor Polynomial constant term (coef₀).  Use [tune()] to
#'   optimize.  (Polynomial specs only.)
#'
#' @return A parsnip `model_spec` object of the corresponding class.
#'
#' @section Engine arguments:
#' The `precondition` argument of [rmspe_lssvr()] is exposed as a non-tunable
#' engine argument. Pass it via [parsnip::set_engine()], e.g.
#' `set_engine("psvr", precondition = "always")`. Default is `"auto"`. See
#' [rmspe_lssvr()] for accepted values and semantics.
#'
#' @examples
#' \dontrun{
#' library(parsnip)
#' spec <- psvr_rmspe_rbf(cost = 1000, rbf_sigma = 1) |>
#'   set_engine("psvr")
#'
#' spec_poly <- psvr_rmspe_poly(cost = 1000, degree = 2, scale_factor = 1) |>
#'   set_engine("psvr")
#'
#' spec_lin <- psvr_rmspe_linear(cost = 1000) |>
#'   set_engine("psvr")
#' }
#'
#' @name psvr_rmspe_specs
#' @export
psvr_rmspe_rbf <- function(mode = "regression", engine = "psvr",
                           cost = NULL, rbf_sigma = NULL) {
  args <- list(
    cost      = rlang::enquo(cost),
    rbf_sigma = rlang::enquo(rbf_sigma)
  )
  parsnip::new_model_spec(
    "psvr_rmspe_rbf_model",
    args                  = args,
    eng_args              = NULL,
    mode                  = mode,
    user_specified_mode   = !missing(mode),
    method                = NULL,
    engine                = engine,
    user_specified_engine = !missing(engine)
  )
}

#' @rdname psvr_rmspe_specs
#' @export
psvr_rmspe_poly <- function(mode = "regression", engine = "psvr",
                            cost = NULL, degree = NULL, scale_factor = NULL) {
  args <- list(
    cost         = rlang::enquo(cost),
    degree       = rlang::enquo(degree),
    scale_factor = rlang::enquo(scale_factor)
  )
  parsnip::new_model_spec(
    "psvr_rmspe_poly_model",
    args                  = args,
    eng_args              = NULL,
    mode                  = mode,
    user_specified_mode   = !missing(mode),
    method                = NULL,
    engine                = engine,
    user_specified_engine = !missing(engine)
  )
}

#' @rdname psvr_rmspe_specs
#' @export
psvr_rmspe_linear <- function(mode = "regression", engine = "psvr",
                              cost = NULL) {
  args <- list(cost = rlang::enquo(cost))
  parsnip::new_model_spec(
    "psvr_rmspe_linear_model",
    args                  = args,
    eng_args              = NULL,
    mode                  = mode,
    user_specified_mode   = !missing(mode),
    method                = NULL,
    engine                = engine,
    user_specified_engine = !missing(engine)
  )
}

#' Parsnip model specs: symmetric LS-SVR with RMSPE loss (Model 4)
#'
#' Create parsnip model specifications for [rmspe_sym_lssvr()] with a fixed
#' kernel type.  The symmetry type is exposed as the tunable `sym_type`
#' argument (`"even"` for a = 1, `"odd"` for a = -1); pass
#' `sym_type = tune()` to let CV select it automatically.
#'
#' @inheritParams psvr_rmspe_specs
#' @param sym_type Symmetry type: `"even"` (default, a = 1) or `"odd"`
#'   (a = -1).  Use [tune()] to optimise over both values during CV.
#'
#' @return A parsnip `model_spec` object of the corresponding class.
#'
#' @section Engine arguments:
#' The `precondition` argument of [rmspe_sym_lssvr()] is exposed as a
#' non-tunable engine argument. Pass it via [parsnip::set_engine()], e.g.
#' `set_engine("psvr", precondition = "always")`. Default is `"auto"`. See
#' [rmspe_sym_lssvr()] for accepted values and semantics.
#'
#' @examples
#' \dontrun{
#' library(parsnip)
#' spec <- psvr_rmspe_sym_rbf(cost = 1000, rbf_sigma = 1) |>
#'   set_engine("psvr")
#'
#' spec_poly <- psvr_rmspe_sym_poly(cost = 1000, degree = 2,
#'                                  scale_factor = 1) |>
#'   set_engine("psvr")
#'
#' spec_lin <- psvr_rmspe_sym_linear(cost = 1000) |>
#'   set_engine("psvr")
#' }
#'
#' @name psvr_rmspe_sym_specs
#' @export
psvr_rmspe_sym_rbf <- function(mode = "regression", engine = "psvr",
                               cost = NULL, rbf_sigma = NULL,
                               sym_type = NULL) {
  args <- list(
    cost      = rlang::enquo(cost),
    rbf_sigma = rlang::enquo(rbf_sigma),
    sym_type  = rlang::enquo(sym_type)
  )
  parsnip::new_model_spec(
    "psvr_rmspe_sym_rbf_model",
    args                  = args,
    eng_args              = NULL,
    mode                  = mode,
    user_specified_mode   = !missing(mode),
    method                = NULL,
    engine                = engine,
    user_specified_engine = !missing(engine)
  )
}

#' @rdname psvr_rmspe_sym_specs
#' @export
psvr_rmspe_sym_poly <- function(mode = "regression", engine = "psvr",
                                cost = NULL, degree = NULL,
                                scale_factor = NULL) {
  args <- list(
    cost         = rlang::enquo(cost),
    degree       = rlang::enquo(degree),
    scale_factor = rlang::enquo(scale_factor)
  )
  parsnip::new_model_spec(
    "psvr_rmspe_sym_poly_model",
    args                  = args,
    eng_args              = NULL,
    mode                  = mode,
    user_specified_mode   = !missing(mode),
    method                = NULL,
    engine                = engine,
    user_specified_engine = !missing(engine)
  )
}

#' @rdname psvr_rmspe_sym_specs
#' @export
psvr_rmspe_sym_linear <- function(mode = "regression", engine = "psvr",
                                  cost = NULL) {
  args <- list(cost = rlang::enquo(cost))
  parsnip::new_model_spec(
    "psvr_rmspe_sym_linear_model",
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
# psvr_update_spec() uses only public parsnip API, avoiding `:::` calls.
# The `parameters` argument is accepted for API compatibility with tune_grid().

psvr_update_spec <- function(object, cls, new_args, fresh, ...) {
  if (fresh) {
    object$args <- new_args
  } else {
    is_null_quo <- vapply(new_args,
                          function(q) rlang::is_quosure(q) && rlang::quo_is_null(q),
                          logical(1L))
    new_args <- new_args[!is_null_quo]
    if (length(new_args) > 0L) object$args[names(new_args)] <- new_args
  }
  eng_dots <- rlang::enquos(...)
  if (length(eng_dots) > 0L) {
    if (is.null(object$eng_args)) object$eng_args <- list()
    if (fresh) object$eng_args <- eng_dots
    else       object$eng_args[names(eng_dots)] <- eng_dots
  }
  parsnip::new_model_spec(
    cls,
    args                  = object$args,
    eng_args              = object$eng_args,
    mode                  = object$mode,
    user_specified_mode   = object$user_specified_mode,
    method                = NULL,
    engine                = object$engine,
    user_specified_engine = object$user_specified_engine
  )
}

#' @export
update.psvr_mape_rbf_model <- function(object, parameters = NULL,
                                       cost = NULL, svm_margin = NULL,
                                       rbf_sigma = NULL,
                                       fresh = FALSE, ...) {
  psvr_update_spec(object, "psvr_mape_rbf_model",
                   list(cost       = rlang::enquo(cost),
                        svm_margin = rlang::enquo(svm_margin),
                        rbf_sigma  = rlang::enquo(rbf_sigma)),
                   fresh, ...)
}

#' @export
update.psvr_mape_poly_model <- function(object, parameters = NULL,
                                        cost = NULL, svm_margin = NULL,
                                        degree = NULL, scale_factor = NULL,
                                        fresh = FALSE, ...) {
  psvr_update_spec(object, "psvr_mape_poly_model",
                   list(cost         = rlang::enquo(cost),
                        svm_margin   = rlang::enquo(svm_margin),
                        degree       = rlang::enquo(degree),
                        scale_factor = rlang::enquo(scale_factor)),
                   fresh, ...)
}

#' @export
update.psvr_mape_linear_model <- function(object, parameters = NULL,
                                          cost = NULL, svm_margin = NULL,
                                          fresh = FALSE, ...) {
  psvr_update_spec(object, "psvr_mape_linear_model",
                   list(cost       = rlang::enquo(cost),
                        svm_margin = rlang::enquo(svm_margin)),
                   fresh, ...)
}

#' @export
update.psvr_mape_sym_rbf_model <- function(object, parameters = NULL,
                                           cost = NULL, svm_margin = NULL,
                                           rbf_sigma = NULL,
                                           sym_type = NULL,
                                           fresh = FALSE, ...) {
  psvr_update_spec(object, "psvr_mape_sym_rbf_model",
                   list(cost       = rlang::enquo(cost),
                        svm_margin = rlang::enquo(svm_margin),
                        rbf_sigma  = rlang::enquo(rbf_sigma),
                        sym_type   = rlang::enquo(sym_type)),
                   fresh, ...)
}

#' @export
update.psvr_mape_sym_poly_model <- function(object, parameters = NULL,
                                            cost = NULL, svm_margin = NULL,
                                            degree = NULL,
                                            scale_factor = NULL,
                                            fresh = FALSE, ...) {
  psvr_update_spec(object, "psvr_mape_sym_poly_model",
                   list(cost         = rlang::enquo(cost),
                        svm_margin   = rlang::enquo(svm_margin),
                        degree       = rlang::enquo(degree),
                        scale_factor = rlang::enquo(scale_factor)),
                   fresh, ...)
}

#' @export
update.psvr_mape_sym_linear_model <- function(object, parameters = NULL,
                                              cost = NULL, svm_margin = NULL,
                                              fresh = FALSE, ...) {
  psvr_update_spec(object, "psvr_mape_sym_linear_model",
                   list(cost       = rlang::enquo(cost),
                        svm_margin = rlang::enquo(svm_margin)),
                   fresh, ...)
}

#' @export
update.psvr_rmspe_rbf_model <- function(object, parameters = NULL,
                                        cost = NULL, rbf_sigma = NULL,
                                        fresh = FALSE, ...) {
  psvr_update_spec(object, "psvr_rmspe_rbf_model",
                   list(cost      = rlang::enquo(cost),
                        rbf_sigma = rlang::enquo(rbf_sigma)),
                   fresh, ...)
}

#' @export
update.psvr_rmspe_poly_model <- function(object, parameters = NULL,
                                         cost = NULL, degree = NULL,
                                         scale_factor = NULL,
                                         fresh = FALSE, ...) {
  psvr_update_spec(object, "psvr_rmspe_poly_model",
                   list(cost         = rlang::enquo(cost),
                        degree       = rlang::enquo(degree),
                        scale_factor = rlang::enquo(scale_factor)),
                   fresh, ...)
}

#' @export
update.psvr_rmspe_linear_model <- function(object, parameters = NULL,
                                           cost = NULL,
                                           fresh = FALSE, ...) {
  psvr_update_spec(object, "psvr_rmspe_linear_model",
                   list(cost = rlang::enquo(cost)),
                   fresh, ...)
}

#' @export
update.psvr_rmspe_sym_rbf_model <- function(object, parameters = NULL,
                                            cost = NULL, rbf_sigma = NULL,
                                            sym_type = NULL,
                                            fresh = FALSE, ...) {
  psvr_update_spec(object, "psvr_rmspe_sym_rbf_model",
                   list(cost      = rlang::enquo(cost),
                        rbf_sigma = rlang::enquo(rbf_sigma),
                        sym_type  = rlang::enquo(sym_type)),
                   fresh, ...)
}

#' @export
update.psvr_rmspe_sym_poly_model <- function(object, parameters = NULL,
                                             cost = NULL, degree = NULL,
                                             scale_factor = NULL,
                                             fresh = FALSE, ...) {
  psvr_update_spec(object, "psvr_rmspe_sym_poly_model",
                   list(cost         = rlang::enquo(cost),
                        degree       = rlang::enquo(degree),
                        scale_factor = rlang::enquo(scale_factor)),
                   fresh, ...)
}

#' @export
update.psvr_rmspe_sym_linear_model <- function(object, parameters = NULL,
                                               cost = NULL,
                                               fresh = FALSE, ...) {
  psvr_update_spec(object, "psvr_rmspe_sym_linear_model",
                   list(cost = rlang::enquo(cost)),
                   fresh, ...)
}


# ---- Engine registration -------------------------------------------------

# Helper: register one model/engine combination with parsnip.
.reg_psvr <- function(model_name, fit_fun, arg_defs, defaults = list()) {
  parsnip::set_new_model(model_name)
  parsnip::set_model_mode(model_name, "regression")
  parsnip::set_model_engine(model_name, mode = "regression", eng = "psvr")
  parsnip::set_dependency(model_name, eng = "psvr", pkg = "psvr")

  for (ad in arg_defs) {
    parsnip::set_model_arg(
      model        = model_name,
      eng          = "psvr",
      parsnip      = ad[[1]],
      original     = ad[[2]],
      func         = ad[[3]],
      has_submodel = FALSE
    )
  }

  parsnip::set_fit(
    model = model_name, eng = "psvr", mode = "regression",
    value = list(
      interface = "matrix",
      protect   = c("x", "y"),
      func      = c(pkg = "psvr", fun = fit_fun),
      defaults  = defaults
    )
  )

  parsnip::set_encoding(
    model   = model_name, eng = "psvr", mode = "regression",
    options = list(
      predictor_indicators = "traditional",
      compute_intercept    = FALSE,
      remove_intercept     = FALSE,
      allow_sparse_x       = FALSE
    )
  )

  parsnip::set_pred(
    model = model_name, eng = "psvr", mode = "regression", type = "numeric",
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

#' Dials parameter for symmetry type
#'
#' Returns a qualitative [dials::new_qual_param()] describing the `sym_type`
#' argument of symmetric psvr model specs.  `"even"` maps to `a = 1L`
#' (standard symmetric kernel); `"odd"` maps to `a = -1L`
#' (anti-symmetric kernel).
#'
#' @return A `qual_param` object.
#' @export
sym_type_param <- function() {
  dials::new_qual_param(
    type   = "character",
    values = c("even", "odd"),
    label  = c(sym_type = "Symmetry type"),
    tags   = "model"
  )
}

# Reusable arg-definition lists (list(parsnip_name, original_name, dials_func))
.A_COST_C     <- list("cost",         "C",            list(pkg = "psvr",  fun = "cost_psvr"))
.A_COST_GAMMA <- list("cost",         "gamma",        list(pkg = "psvr",  fun = "cost_psvr"))
.A_MARGIN     <- list("svm_margin",   "eps",          list(pkg = "psvr",  fun = "margin_percentage"))
.A_SIGMA      <- list("rbf_sigma",    "rbf_sigma",    list(pkg = "psvr",  fun = "rbf_sigma_psvr"))
.A_DEGREE     <- list("degree",       "degree",       list(pkg = "dials", fun = "degree"))
.A_SCALE      <- list("scale_factor", "scale_factor", list(pkg = "dials", fun = "scale_factor"))
.A_SYM_TYPE   <- list("sym_type",     "sym_type",     list(pkg = "psvr",  fun = "sym_type_param"))

make_psvr_engines <- function() {
  # Skip if already registered — parsnip's env persists across devtools reloads.
  if ("psvr_mape_rbf_model" %in% parsnip::get_from_env("models")) {
    return(invisible(NULL))
  }

  # ---- Model 1: epsilon-SVR with MAPE ----
  .reg_psvr("psvr_mape_rbf_model",    "psvr_mape_rbf_fit",
            list(.A_COST_C, .A_MARGIN, .A_SIGMA))
  .reg_psvr("psvr_mape_poly_model",   "psvr_mape_poly_fit",
            list(.A_COST_C, .A_MARGIN, .A_DEGREE, .A_SCALE))
  .reg_psvr("psvr_mape_linear_model", "psvr_mape_linear_fit",
            list(.A_COST_C, .A_MARGIN))

  # ---- Model 2: symmetric epsilon-SVR with MAPE ----
  .reg_psvr("psvr_mape_sym_rbf_model",    "psvr_mape_sym_rbf_fit",
            list(.A_COST_C, .A_MARGIN, .A_SIGMA, .A_SYM_TYPE))
  .reg_psvr("psvr_mape_sym_poly_model",   "psvr_mape_sym_poly_fit",
            list(.A_COST_C, .A_MARGIN, .A_DEGREE, .A_SCALE), defaults = list(a = 1L))
  .reg_psvr("psvr_mape_sym_linear_model", "psvr_mape_sym_linear_fit",
            list(.A_COST_C, .A_MARGIN), defaults = list(a = 1L))

  # ---- Model 3: LS-SVR with RMSPE ----
  .reg_psvr("psvr_rmspe_rbf_model",    "psvr_rmspe_rbf_fit",
            list(.A_COST_GAMMA, .A_SIGMA),
            defaults = list(precondition = "auto"))
  .reg_psvr("psvr_rmspe_poly_model",   "psvr_rmspe_poly_fit",
            list(.A_COST_GAMMA, .A_DEGREE, .A_SCALE),
            defaults = list(precondition = "auto"))
  .reg_psvr("psvr_rmspe_linear_model", "psvr_rmspe_linear_fit",
            list(.A_COST_GAMMA),
            defaults = list(precondition = "auto"))

  # ---- Model 4: symmetric LS-SVR with RMSPE ----
  .reg_psvr("psvr_rmspe_sym_rbf_model",    "psvr_rmspe_sym_rbf_fit",
            list(.A_COST_GAMMA, .A_SIGMA, .A_SYM_TYPE),
            defaults = list(precondition = "auto"))
  .reg_psvr("psvr_rmspe_sym_poly_model",   "psvr_rmspe_sym_poly_fit",
            list(.A_COST_GAMMA, .A_DEGREE, .A_SCALE),
            defaults = list(a = 1L, precondition = "auto"))
  .reg_psvr("psvr_rmspe_sym_linear_model", "psvr_rmspe_sym_linear_fit",
            list(.A_COST_GAMMA),
            defaults = list(a = 1L, precondition = "auto"))
}


# ---- Package hook --------------------------------------------------------

.onLoad <- function(libname, pkgname) {
  if (requireNamespace("parsnip", quietly = TRUE)) {
    make_psvr_engines()
  }
}
