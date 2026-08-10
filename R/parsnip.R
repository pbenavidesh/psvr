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


# ---- Symmetry-type translation -------------------------------------------
# Maps the `sym_type` model argument onto the solver's integer `a`. Callers
# branch on "none" BEFORE reaching this helper (that level selects a different
# fitter, not a different `a`), so "none" arriving here is a caller bug and is
# reported as one rather than silently treated as "odd".
#
# The strict switch() replaces the earlier `if (sym_type == "even") 1L else -1L`
# form, which mapped every non-"even" string — including typos — to -1L.
.sym_type_to_a <- function(sym_type) {
  if (!is.character(sym_type) || length(sym_type) != 1L)
    stop("`sym_type` must be a single string, one of \"none\", \"even\", ",
         "\"odd\".", call. = FALSE)
  switch(sym_type,
         even = 1L,
         odd  = -1L,
         stop("`sym_type` must be one of \"none\", \"even\", \"odd\"; got \"",
              sym_type, "\".", call. = FALSE))
}


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
#' @param sym_type Symmetry type. `"none"` (the default) dispatches to the
#'   non-symmetric fitter; `"even"` and `"odd"` dispatch to the symmetric
#'   fitter with `a = 1L` and `a = -1L` respectively.
#' @param tol Solver convergence tolerance for the SMO loop. Default `1e-3`.
#' @param max_iter Maximum SMO iterations. Default `100000L`. The solver
#'   emits a `warning()` and returns `solver_meta$converged = FALSE` if it
#'   does not converge within `max_iter`.
#' @param precondition Optional symmetric rescaling preconditioner for the
#'   RMSPE LS-SVR fitters. See [psvr()] for accepted values and semantics.
#' @return A fitted model object of the legacy S3 class matching the wrapper's
#'   model family. These are the pre-[psvr()] object shapes, returned
#'   unmodified from the internal fitter; they are **not** `psvr_fit` objects.
#'   **Which class is returned depends on `sym_type`**, since each wrapper
#'   dispatches to the symmetric or non-symmetric fitter.
#'
#'   The MAPE wrappers (`psvr_mape_rbf_fit()`, `psvr_mape_poly_fit()`,
#'   `psvr_mape_linear_fit()`) with `sym_type = "none"` return an object of
#'   class `"psvr_mape"`: a list with `beta` (support-vector dual differences),
#'   `alpha` and `alpha_star` (length-`N` pre-pruning duals, retained for warm
#'   starts), `b`, `X_sv`, `y_sv`, `y_train`, `fitted_values`, `kernel`, `C`,
#'   `eps`, `n_train`, `p_train`, `iterations`, `converged`, and `block_k4`.
#'   With `sym_type = "even"` or `"odd"` they return class `"psvr_mape_sym"`:
#'   the same components plus `a` (the symmetry type) and `spectral`
#'   (Algorithm 2 diagnostics).
#'
#'   The RMSPE wrappers (`psvr_rmspe_rbf_fit()`, `psvr_rmspe_poly_fit()`,
#'   `psvr_rmspe_linear_fit()`) with `sym_type = "none"` return class
#'   `"psvr_rmspe"`: a list with `alpha`, `b`, `X_train`, `y_train`,
#'   `fitted_values`, `kernel`, `gamma`, `n_train`, `p_train`, and
#'   `precondition_applied`. With `sym_type = "even"` or `"odd"` they return
#'   class `"psvr_rmspe_sym"`: the same components plus `a`.
#' @name psvr-fit-wrappers
#' @keywords internal
NULL

#' @rdname psvr-fit-wrappers
#' @keywords internal
#' @export
psvr_mape_rbf_fit <- function(x, y, C, eps, rbf_sigma = 1,
                              sym_type = "none",
                              tol = 1e-3, max_iter = 100000L) {
  K <- make_kernel("rbf", sigma = rbf_sigma)
  if (identical(sym_type, "none")) {
    .fit_mape(X = x, y = y, kernel = K, C = C, eps = eps,
              tol = tol, max_iter = max_iter)
  } else {
    .fit_mape_sym(X = x, y = y, kernel = K, C = C, eps = eps,
                  a = .sym_type_to_a(sym_type),
                  tol = tol, max_iter = max_iter)
  }
}

#' @rdname psvr-fit-wrappers
#' @keywords internal
#' @export
psvr_mape_poly_fit <- function(x, y, C, eps, degree = 3L, scale_factor = 1,
                               sym_type = "none",
                               tol = 1e-3, max_iter = 100000L) {
  K <- make_kernel("polynomial", degree = degree, coef0 = scale_factor)
  if (identical(sym_type, "none")) {
    .fit_mape(X = x, y = y, kernel = K, C = C, eps = eps,
              tol = tol, max_iter = max_iter)
  } else {
    .fit_mape_sym(X = x, y = y, kernel = K, C = C, eps = eps,
                  a = .sym_type_to_a(sym_type),
                  tol = tol, max_iter = max_iter)
  }
}

#' @rdname psvr-fit-wrappers
#' @keywords internal
#' @export
psvr_mape_linear_fit <- function(x, y, C, eps, sym_type = "none",
                                 tol = 1e-3, max_iter = 100000L) {
  K <- make_kernel("linear")
  if (identical(sym_type, "none")) {
    .fit_mape(X = x, y = y, kernel = K, C = C, eps = eps,
              tol = tol, max_iter = max_iter)
  } else {
    .fit_mape_sym(X = x, y = y, kernel = K, C = C, eps = eps,
                  a = .sym_type_to_a(sym_type),
                  tol = tol, max_iter = max_iter)
  }
}

#' @rdname psvr-fit-wrappers
#' @keywords internal
#' @export
psvr_rmspe_rbf_fit <- function(x, y, gamma, rbf_sigma = 1,
                               sym_type = "none",
                               precondition = "auto") {
  K <- make_kernel("rbf", sigma = rbf_sigma)
  if (identical(sym_type, "none")) {
    .fit_rmspe(X = x, y = y, kernel = K,
               gamma = gamma, precondition = precondition)
  } else {
    .fit_rmspe_sym(X = x, y = y, kernel = K,
                   gamma = gamma, a = .sym_type_to_a(sym_type),
                   precondition = precondition)
  }
}

#' @rdname psvr-fit-wrappers
#' @keywords internal
#' @export
psvr_rmspe_poly_fit <- function(x, y, gamma, degree = 3L, scale_factor = 1,
                                sym_type = "none",
                                precondition = "auto") {
  K <- make_kernel("polynomial", degree = degree, coef0 = scale_factor)
  if (identical(sym_type, "none")) {
    .fit_rmspe(X = x, y = y, kernel = K,
               gamma = gamma, precondition = precondition)
  } else {
    .fit_rmspe_sym(X = x, y = y, kernel = K,
                   gamma = gamma, a = .sym_type_to_a(sym_type),
                   precondition = precondition)
  }
}

#' @rdname psvr-fit-wrappers
#' @keywords internal
#' @export
psvr_rmspe_linear_fit <- function(x, y, gamma, sym_type = "none",
                                  precondition = "auto") {
  K <- make_kernel("linear")
  if (identical(sym_type, "none")) {
    .fit_rmspe(X = x, y = y, kernel = K,
               gamma = gamma, precondition = precondition)
  } else {
    .fit_rmspe_sym(X = x, y = y, kernel = K,
                   gamma = gamma, a = .sym_type_to_a(sym_type),
                   precondition = precondition)
  }
}


# ---- Constructors --------------------------------------------------------

#' Parsnip model specs: epsilon-SVR with MAPE loss (Model 1)
#'
#' Create parsnip model specifications for [psvr()] with a fixed kernel
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
#' @param sym_type Symmetry type: `"none"` (default) fits the non-symmetric
#'   ε-SVR of Model 1; `"even"` (a = 1) and `"odd"` (a = -1) fit the
#'   symmetric ε-SVR of Model 2.  Use [tune()] to optimise over the levels
#'   during CV; see [sym_type_param()] to restrict which levels are searched.
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
#'
#' # Symmetric epsilon-SVR (Model 2) via the sym_type argument:
#' spec_sym <- psvr_mape_rbf(cost = 10, svm_margin = 1, rbf_sigma = 1,
#'                           sym_type = "even") |>
#'   set_engine("psvr")
#' }
#'
#' @name psvr_mape_specs
#' @export
psvr_mape_rbf <- function(mode = "regression", engine = "psvr",
                          cost = NULL, svm_margin = NULL, rbf_sigma = NULL,
                          sym_type = NULL) {
  args <- list(
    cost       = rlang::enquo(cost),
    svm_margin = rlang::enquo(svm_margin),
    rbf_sigma  = rlang::enquo(rbf_sigma),
    sym_type   = rlang::enquo(sym_type)
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
                           degree = NULL, scale_factor = NULL,
                           sym_type = NULL) {
  args <- list(
    cost         = rlang::enquo(cost),
    svm_margin   = rlang::enquo(svm_margin),
    degree       = rlang::enquo(degree),
    scale_factor = rlang::enquo(scale_factor),
    sym_type     = rlang::enquo(sym_type)
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
                             cost = NULL, svm_margin = NULL,
                             sym_type = NULL) {
  args <- list(
    cost       = rlang::enquo(cost),
    svm_margin = rlang::enquo(svm_margin),
    sym_type   = rlang::enquo(sym_type)
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

#' Parsnip model specs: LS-SVR with RMSPE loss (Model 3)
#'
#' Create parsnip model specifications for [psvr()] with a fixed kernel
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
#' @param sym_type Symmetry type: `"none"` (default) fits the non-symmetric
#'   LS-SVR of Model 3; `"even"` (a = 1) and `"odd"` (a = -1) fit the
#'   symmetric LS-SVR of Model 4.  Use [tune()] to optimise over the levels
#'   during CV; see [sym_type_param()] to restrict which levels are searched.
#'
#' @return A parsnip `model_spec` object of the corresponding class.
#'
#' @section Engine arguments:
#' The `precondition` argument of [psvr()] is exposed as a non-tunable
#' engine argument. Pass it via [parsnip::set_engine()], e.g.
#' `set_engine("psvr", precondition = "always")`. Default is `"auto"`. See
#' [psvr()] for accepted values and semantics.
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
#'
#' # Symmetric LS-SVR (Model 4) via the sym_type argument:
#' spec_sym <- psvr_rmspe_rbf(cost = 1000, rbf_sigma = 1,
#'                            sym_type = "even") |>
#'   set_engine("psvr")
#' }
#'
#' @name psvr_rmspe_specs
#' @export
psvr_rmspe_rbf <- function(mode = "regression", engine = "psvr",
                           cost = NULL, rbf_sigma = NULL, sym_type = NULL) {
  args <- list(
    cost      = rlang::enquo(cost),
    rbf_sigma = rlang::enquo(rbf_sigma),
    sym_type  = rlang::enquo(sym_type)
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
                            cost = NULL, degree = NULL, scale_factor = NULL,
                            sym_type = NULL) {
  args <- list(
    cost         = rlang::enquo(cost),
    degree       = rlang::enquo(degree),
    scale_factor = rlang::enquo(scale_factor),
    sym_type     = rlang::enquo(sym_type)
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
                              cost = NULL, sym_type = NULL) {
  args <- list(
    cost     = rlang::enquo(cost),
    sym_type = rlang::enquo(sym_type)
  )
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
                                       rbf_sigma = NULL, sym_type = NULL,
                                       fresh = FALSE, ...) {
  psvr_update_spec(object, "psvr_mape_rbf_model",
                   list(cost       = rlang::enquo(cost),
                        svm_margin = rlang::enquo(svm_margin),
                        rbf_sigma  = rlang::enquo(rbf_sigma),
                        sym_type   = rlang::enquo(sym_type)),
                   fresh, ...)
}

#' @export
update.psvr_mape_poly_model <- function(object, parameters = NULL,
                                        cost = NULL, svm_margin = NULL,
                                        degree = NULL, scale_factor = NULL,
                                        sym_type = NULL,
                                        fresh = FALSE, ...) {
  psvr_update_spec(object, "psvr_mape_poly_model",
                   list(cost         = rlang::enquo(cost),
                        svm_margin   = rlang::enquo(svm_margin),
                        degree       = rlang::enquo(degree),
                        scale_factor = rlang::enquo(scale_factor),
                        sym_type     = rlang::enquo(sym_type)),
                   fresh, ...)
}

#' @export
update.psvr_mape_linear_model <- function(object, parameters = NULL,
                                          cost = NULL, svm_margin = NULL,
                                          sym_type = NULL,
                                          fresh = FALSE, ...) {
  psvr_update_spec(object, "psvr_mape_linear_model",
                   list(cost       = rlang::enquo(cost),
                        svm_margin = rlang::enquo(svm_margin),
                        sym_type   = rlang::enquo(sym_type)),
                   fresh, ...)
}

#' @export
update.psvr_rmspe_rbf_model <- function(object, parameters = NULL,
                                        cost = NULL, rbf_sigma = NULL,
                                        sym_type = NULL,
                                        fresh = FALSE, ...) {
  psvr_update_spec(object, "psvr_rmspe_rbf_model",
                   list(cost      = rlang::enquo(cost),
                        rbf_sigma = rlang::enquo(rbf_sigma),
                        sym_type  = rlang::enquo(sym_type)),
                   fresh, ...)
}

#' @export
update.psvr_rmspe_poly_model <- function(object, parameters = NULL,
                                         cost = NULL, degree = NULL,
                                         scale_factor = NULL, sym_type = NULL,
                                         fresh = FALSE, ...) {
  psvr_update_spec(object, "psvr_rmspe_poly_model",
                   list(cost         = rlang::enquo(cost),
                        degree       = rlang::enquo(degree),
                        scale_factor = rlang::enquo(scale_factor),
                        sym_type     = rlang::enquo(sym_type)),
                   fresh, ...)
}

#' @export
update.psvr_rmspe_linear_model <- function(object, parameters = NULL,
                                           cost = NULL, sym_type = NULL,
                                           fresh = FALSE, ...) {
  psvr_update_spec(object, "psvr_rmspe_linear_model",
                   list(cost     = rlang::enquo(cost),
                        sym_type = rlang::enquo(sym_type)),
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
#' argument of the psvr model specs.  `"none"` fits the non-symmetric model;
#' `"even"` maps to `a = 1L` (standard symmetric kernel); `"odd"` maps to
#' `a = -1L` (anti-symmetric kernel).
#'
#' @param values Character vector of levels to search over.  Any subset of
#'   `c("none", "even", "odd")`; defaults to all three.  Pass
#'   `values = c("even", "odd")` to tune over the symmetric models only,
#'   which reproduces the two-level grid offered before psvr 0.0.2.9011.
#'
#' @return A `qual_param` object.
#'
#' @examples
#' sym_type_param()
#' sym_type_param(values = c("even", "odd"))
#'
#' @export
sym_type_param <- function(values = c("none", "even", "odd")) {
  values <- rlang::arg_match(values, c("none", "even", "odd"), multiple = TRUE)
  dials::new_qual_param(
    type   = "character",
    values = values,
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

  # ---- Model 1: epsilon-SVR with MAPE (sym_type = "none" by default) ----
  .reg_psvr("psvr_mape_rbf_model",    "psvr_mape_rbf_fit",
            list(.A_COST_C, .A_MARGIN, .A_SIGMA, .A_SYM_TYPE))
  .reg_psvr("psvr_mape_poly_model",   "psvr_mape_poly_fit",
            list(.A_COST_C, .A_MARGIN, .A_DEGREE, .A_SCALE, .A_SYM_TYPE))
  .reg_psvr("psvr_mape_linear_model", "psvr_mape_linear_fit",
            list(.A_COST_C, .A_MARGIN, .A_SYM_TYPE))

  # ---- Model 3: LS-SVR with RMSPE (sym_type = "none" by default) ----
  .reg_psvr("psvr_rmspe_rbf_model",    "psvr_rmspe_rbf_fit",
            list(.A_COST_GAMMA, .A_SIGMA, .A_SYM_TYPE),
            defaults = list(precondition = "auto"))
  .reg_psvr("psvr_rmspe_poly_model",   "psvr_rmspe_poly_fit",
            list(.A_COST_GAMMA, .A_DEGREE, .A_SCALE, .A_SYM_TYPE),
            defaults = list(precondition = "auto"))
  .reg_psvr("psvr_rmspe_linear_model", "psvr_rmspe_linear_fit",
            list(.A_COST_GAMMA, .A_SYM_TYPE),
            defaults = list(precondition = "auto"))
}


# ---- Package hook --------------------------------------------------------

.onLoad <- function(libname, pkgname) {
  if (requireNamespace("parsnip", quietly = TRUE)) {
    make_psvr_engines()
  }
}
