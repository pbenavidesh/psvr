# Public entry point for the epsilon-SVR / MAPE family (Models 1 and 2).
#
# ASCII-only by design, including the roxygen: non-ASCII on a #' line reaches
# man/*.Rd and breaks the PDF manual, which is a hard CRAN blocker (see
# PSVR_STATUS.md 2.20). Mathematical notation uses Rd \eqn{} / \deqn{} in the
# two-argument (LaTeX, ASCII) form.

#' Fit an epsilon-SVR with MAPE loss
#'
#' Fits the percentage-error epsilon-SVR of the paper: Model 1 when
#' `sym_type = "none"`, and the symmetric-kernel Model 2 when `sym_type` is
#' `"even"` or `"odd"`. The dual is a quadratic program with the
#' sample-dependent box \eqn{|\beta_k| \le 100C/y_k}{|beta_k| <= 100*C/y_k}
#' and \eqn{\sum_k \beta_k = 0}{sum_k beta_k = 0}, solved by the built-in SMO
#' loop or by `osqp`.
#'
#' For the least-squares / RMSPE family (Models 3 and 4) see [psvr_rmspe()].
#' The two are deliberately separate functions: they share no solver, no dual
#' structure and no hyperparameter search space, so a single signature would
#' make most of its own arguments conditional. The name `psvr()` is reserved
#' for a future automatic-selection front end and is **not** a synonym for
#' either.
#'
#' @section Choosing a solver:
#' `solver = "smo"` (the default) uses the built-in sequential minimal
#' optimisation loop. It does **not** reliably converge within `max_iter` on
#' linear and polynomial kernels; `converged` is `FALSE` and `iterations`
#' reaches the cap. The RBF kernel is unaffected. Prefer `solver = "osqp"`
#' for linear and polynomial kernels when accuracy matters.
#'
#' @param X Numeric matrix of training inputs, one observation per row
#'   (\eqn{N \times p}{N x p}).
#' @param y Numeric vector of training targets, length \eqn{N}{N}. Must satisfy
#'   \eqn{y_k > 0}{y_k > 0} for every \eqn{k}{k}; percentage-error loss is
#'   undefined otherwise, and this is checked rather than coerced.
#' @param sym_type Symmetry type, one of `"none"` (default), `"even"` or
#'   `"odd"`. Maps onto the symmetry parameter \eqn{a}{a} of the paper:
#'   `"none"` fits Model 1 and imposes no symmetry constraint; `"even"` sets
#'   \eqn{a = +1}{a = +1}, enforcing \eqn{f(x) = f(-x)}{f(x) = f(-x)}; `"odd"`
#'   sets \eqn{a = -1}{a = -1}, enforcing \eqn{f(x) = -f(-x)}{f(x) = -f(-x)}.
#'   This is the same vocabulary as the `sym_type` argument of the parsnip
#'   specifications, so the two public surfaces agree. The symmetric variants
#'   require a kernel satisfying Assumption 3 of the paper -- see
#'   [make_kernel()].
#' @param kernel A kernel function created by [make_kernel()].
#' @param C Regularization parameter, \eqn{C > 0}{C > 0}. Required.
#' @param eps Insensitivity tube half-width in percentage units,
#'   \eqn{\epsilon \ge 0}{eps >= 0}. Required.
#' @param solver Backend for the dual quadratic program, `"smo"` (default) or
#'   `"osqp"`. See the section above.
#' @param tol Numerical tolerance, default `1e-3`. **Its meaning depends on
#'   `solver`.** Under `"smo"` it is the convergence tolerance of the SMO loop,
#'   and tighter values produce more iterations. Under `"osqp"` the solver runs
#'   at its own fixed tolerances and `tol` is used only afterwards, as the
#'   threshold below which a dual variable counts as zero when identifying free,
#'   saturated and support-vector sets.
#' @param max_iter Maximum SMO iterations, default `100000L`. The solver emits a
#'   `warning()` and returns `converged = FALSE` if it does not converge within
#'   `max_iter`. Ignored for `solver = "osqp"`.
#' @param alpha_init,alpha_star_init Optional warm-start vectors for the SMO
#'   solver, each a finite numeric vector of length \eqn{N}{N}. Projected onto
#'   \eqn{\sum_k (\alpha_k - \alpha^*_k) = 0}{sum_k (alpha_k - alpha_star_k) = 0}
#'   intersected with the per-sample box \eqn{[0, 100C/y_k]}{[0, 100*C/y_k]}
#'   before the solve. `NULL` (default) cold-starts. [psvr_cv()] manages these
#'   automatically across folds.
#' @param warm_start_check Logical; if `TRUE` (default), validate
#'   post-projection feasibility and `stop()` on violation. A surviving equality
#'   residual is fatal rather than cosmetic: SMO conserves
#'   \eqn{\sum_k (\alpha_k - \alpha^*_k)}{sum_k (alpha_k - alpha_star_k)}, so an
#'   infeasible start is carried through to the returned solution.
#' @param block_k4_enabled Logical; if `TRUE` (default), enable the block-k=4
#'   SMO inner loop. Each outer iteration may select a second working pair and
#'   apply a two-dimensional joint update when the descent-guaranteed decoupling
#'   criterion holds. `FALSE` restores the k=2 behaviour bit-identically.
#' @param engine One of `"rcpp"` (default) or `"r"`. Selects the SMO backend:
#'   the C++ core, or the R reference implementation. Both produce
#'   bit-identical results on the development toolchain; `"r"` is retained as
#'   the reference and will be deprecated in 0.0.4.0 and removed in 0.1.0.
#' @param ... Must be empty. Present only so that `alpha_couple`,
#'   `precomputed_Omega` and `precomputed_Omega_s` must be matched by their
#'   exact names rather than by position or partial matching -- without it
#'   `precomputed_Omega` would be a partial-match prefix of
#'   `precomputed_Omega_s`. Passing anything here is an error, which is how a
#'   mistyped argument name is caught.
#' @param alpha_couple Numeric in \eqn{[0, 1]}{[0, 1]}, default `0.5`. Coupling
#'   penalty in the second-pair selection score
#'   \eqn{\mathrm{gain} \times (1 - \alpha_{\mathrm{couple}} \cdot
#'   \mathrm{coupling})}{gain * (1 - alpha_couple * coupling)}. Exposed for
#'   empirical tuning; rarely needs adjustment. Ignored when
#'   `block_k4_enabled = FALSE`.
#' @param precomputed_Omega,precomputed_Omega_s INTERNAL -- used by [psvr_cv()]
#'   to share one full-dataset kernel matrix across folds. Users should not set
#'   these. `precomputed_Omega` applies when `sym_type = "none"`,
#'   `precomputed_Omega_s` otherwise.
#'
#' @return For `sym_type = "none"`, an object of class `"psvr_mape"`: a list
#'   with components `beta` (the pruned dual differences
#'   \eqn{\beta = \alpha - \alpha^*}{beta = alpha - alpha_star} over the
#'   support-vector indices, used by `predict()`), `alpha` and `alpha_star`
#'   (the length-\eqn{N}{N} pre-pruning duals, retained for warm starts), `b`,
#'   `X_sv`, `y_sv`, `y_train`, `fitted_values`, `kernel`, `C`, `eps`,
#'   `n_train`, `p_train`, `iterations`, `converged` and `block_k4`.
#'
#'   For `sym_type = "even"` or `"odd"`, an object of class `"psvr_mape_sym"`:
#'   the same components plus `a` (the symmetry parameter) and `spectral`
#'   (adaptive spectral-shift diagnostics).
#'
#'   Methods are available for [predict()], [print()], [coef()], [summary()],
#'   [fitted()] and [residuals()].
#'
#' @seealso [psvr_rmspe()] for the LS-SVR / RMSPE family, [psvr_cv()] for
#'   cross-validation with warm-start carryover, [make_kernel()] for kernels.
#'
#' @examples
#' set.seed(1)
#' X <- matrix(rnorm(40), 20, 2)
#' y <- rlnorm(20)
#' K <- make_kernel("rbf", sigma = 1)
#'
#' fit <- psvr_mape(X, y, kernel = K, C = 10, eps = 5)
#' predict(fit, X[1:3, , drop = FALSE])
#'
#' # Even-symmetric variant (Model 2): f(x) = f(-x).
#' fit_sym <- psvr_mape(X, y, sym_type = "even", kernel = K, C = 10, eps = 5)
#' predict(fit_sym, X[1:3, , drop = FALSE])
#'
#' @export
psvr_mape <- function(X, y,
                      sym_type = c("none", "even", "odd"),
                      kernel,
                      C, eps,
                      solver   = c("smo", "osqp"),
                      tol      = 1e-3,
                      max_iter = 100000L,
                      alpha_init       = NULL,
                      alpha_star_init  = NULL,
                      warm_start_check = TRUE,
                      block_k4_enabled = TRUE,
                      engine           = c("rcpp", "r"),
                      ...,
                      alpha_couple        = 0.5,
                      precomputed_Omega   = NULL,
                      precomputed_Omega_s = NULL) {

  # `reg` was a formal of the superseded psvr() whose only behaviour was to error
  # on any non-NULL value: a not-implemented placeholder, never a validity
  # guard. It is gone, but bare check_dots_empty() would only say `reg` is
  # unknown -- not that the feature is unimplemented. Keep the explanation.
  if ("reg" %in% ...names())
    stop("`reg` is not an argument of `psvr_mape()`. The extended Lagrangian ",
         "(elastic-net) penalty is not implemented -- see PSVR_STATUS.md. When ",
         "it lands it will add an argument, not restore this one.",
         call. = FALSE)

  # `...` exists only to force exact-name matching on the three internal
  # arguments that follow it. Nothing may be passed through it, so reject
  # unknown arguments instead of swallowing typos.
  rlang::check_dots_empty()

  sym_type <- match.arg(sym_type)
  solver   <- match.arg(solver)
  engine   <- match.arg(engine)

  # missing() only works in the frame that owns the formal, so presence checks
  # cannot be delegated to a validator.
  if (missing(kernel))
    stop("`kernel` is required; build one with `make_kernel()`.", call. = FALSE)
  if (missing(C))   stop("`C` is required.",   call. = FALSE)
  if (missing(eps)) stop("`eps` is required.", call. = FALSE)

  X <- as.matrix(X)
  y <- as.numeric(y)

  .validate_mape_inputs(X, alpha_init, alpha_star_init)

  if (identical(sym_type, "none")) {
    .fit_mape(X, y, kernel = kernel, C = C, eps = eps,
              solver = solver, tol = tol, max_iter = max_iter,
              alpha_init = alpha_init,
              alpha_star_init = alpha_star_init,
              warm_start_check = warm_start_check,
              precomputed_Omega = precomputed_Omega,
              block_k4_enabled = block_k4_enabled,
              alpha_couple = alpha_couple,
              engine = engine)
  } else {
    .fit_mape_sym(X, y, kernel = kernel, C = C, eps = eps,
                  a = .sym_type_to_a(sym_type),
                  solver = solver, tol = tol, max_iter = max_iter,
                  alpha_init = alpha_init,
                  alpha_star_init = alpha_star_init,
                  warm_start_check = warm_start_check,
                  precomputed_Omega_s = precomputed_Omega_s,
                  block_k4_enabled = block_k4_enabled,
                  alpha_couple = alpha_couple,
                  engine = engine)
  }
}
