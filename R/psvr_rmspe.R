# Public entry point for the LS-SVR / RMSPE family (Models 3 and 4).
#
# ASCII-only by design, including the roxygen -- see the note at the head of
# R/psvr_mape.R.

#' Fit a least-squares SVR with RMSPE loss
#'
#' Fits the percentage-error LS-SVR of the paper: Model 3 when
#' `sym_type = "none"`, and the symmetric-kernel Model 4 when `sym_type` is
#' `"even"` or `"odd"`. There is no quadratic program and no sparsity: the fit
#' is a single solve of the \eqn{(N+1) \times (N+1)}{(N+1) x (N+1)} augmented
#' linear system
#' \deqn{\begin{pmatrix} 0 & 1^{\top} \\ 1 & \Omega + Y_{\Gamma}\end{pmatrix}
#'       \begin{pmatrix} b \\ \alpha \end{pmatrix} =
#'       \begin{pmatrix} 0 \\ y \end{pmatrix}}{%
#'       [0, 1'; 1, Omega + Y_Gamma] [b; alpha] = [0; y]}
#' with \eqn{Y_{\Gamma} = \mathrm{diag}(y_1^2/\Gamma, \ldots,
#' y_N^2/\Gamma)}{Y_Gamma = diag(y_1^2/Gamma, ..., y_N^2/Gamma)}, and
#' \eqn{\Omega_s}{Omega_s} replacing \eqn{\Omega}{Omega} in the symmetric case.
#' Every training point contributes to the prediction.
#'
#' For the epsilon-SVR / MAPE family (Models 1 and 2) see [psvr_mape()]. The two
#' are deliberately separate functions: they share no solver, no dual structure
#' and no hyperparameter search space. The name `psvr()` is reserved for a
#' future automatic-selection front end and is **not** a synonym for either.
#'
#' @param X Numeric matrix of training inputs, one observation per row
#'   (\eqn{N \times p}{N x p}).
#' @param y Numeric vector of training targets, length \eqn{N}{N}. Must satisfy
#'   \eqn{y_k > 0}{y_k > 0} for every \eqn{k}{k}; percentage-error loss is
#'   undefined otherwise, and this is checked rather than coerced.
#' @param sym_type Symmetry type, one of `"none"` (default), `"even"` or
#'   `"odd"`. Maps onto the symmetry parameter \eqn{a}{a} of the paper:
#'   `"none"` fits Model 3 and imposes no symmetry constraint; `"even"` sets
#'   \eqn{a = +1}{a = +1}, enforcing \eqn{f(x) = f(-x)}{f(x) = f(-x)}; `"odd"`
#'   sets \eqn{a = -1}{a = -1}, enforcing \eqn{f(x) = -f(-x)}{f(x) = -f(-x)}.
#'   This is the same vocabulary as the `sym_type` argument of the parsnip
#'   specifications, so the two public surfaces agree. The symmetric variants
#'   require a kernel satisfying Assumption 3 of the paper -- see
#'   [make_kernel()].
#' @param kernel A kernel function created by [make_kernel()].
#' @param gamma Regularization parameter \eqn{\Gamma > 0}{Gamma > 0}. Required.
#'   Larger values weight the squared percentage residuals more heavily against
#'   the norm penalty.
#' @param precondition One of `"auto"` (default), `"always"`, `"never"`, or a
#'   positive numeric threshold. Controls the symmetric rescaling of Remark 17.
#'   `"auto"` applies it when the target ratio
#'   \eqn{\max(y)/\min(y)}{max(y)/min(y)} exceeds 10; a numeric value sets that
#'   threshold explicitly. Whether it fired is reported in
#'   `fit$precondition_applied`.
#' @param ... Must be empty. Passing anything here is an error, which is how a
#'   mistyped argument name is caught.
#'
#' @return For `sym_type = "none"`, an object of class `"psvr_rmspe"`: a list
#'   with components `alpha` (the length-\eqn{N}{N} multipliers), `b`,
#'   `X_train`, `y_train`, `fitted_values`, `kernel`, `gamma`, `n_train`,
#'   `p_train` and `precondition_applied`.
#'
#'   For `sym_type = "even"` or `"odd"`, an object of class `"psvr_rmspe_sym"`:
#'   the same components plus `a` (the symmetry parameter).
#'
#'   Methods are available for [predict()], [print()], [coef()], [summary()],
#'   [fitted()] and [residuals()]. Note that `coef()` returns three components
#'   here (`alpha`, `b`, `support_data`) against five for the MAPE classes:
#'   LS-SVR has no `alpha_star` and no pruned `beta`, and the absent components
#'   are not materialised as `NULL`.
#'
#' @seealso [psvr_mape()] for the epsilon-SVR / MAPE family, [make_kernel()]
#'   for kernels.
#'
#' @examples
#' set.seed(1)
#' X <- matrix(rnorm(40), 20, 2)
#' y <- rlnorm(20)
#' K <- make_kernel("rbf", sigma = 1)
#'
#' fit <- psvr_rmspe(X, y, kernel = K, gamma = 100)
#' predict(fit, X[1:3, , drop = FALSE])
#'
#' # Even-symmetric variant (Model 4): f(x) = f(-x).
#' fit_sym <- psvr_rmspe(X, y, sym_type = "even", kernel = K, gamma = 100)
#' predict(fit_sym, X[1:3, , drop = FALSE])
#'
#' @export
psvr_rmspe <- function(X, y,
                       sym_type = c("none", "even", "odd"),
                       kernel,
                       gamma,
                       precondition = "auto",
                       ...) {

  # LS-SVR is a single linear-system solve: there is no SMO state to carry
  # over, so warm starts are not a missing feature but a meaningless one. The
  # superseded psvr() said so explicitly and a test pins the wording; keep
  # since check_dots_empty() alone would report only that the name is unknown.
  if (any(c("alpha_init", "alpha_star_init") %in% ...names()))
    stop("Warm-start is not supported for the LS-SVR / RMSPE family (it is a ",
         "single linear-system solve; there is no SMO state to carry over). ",
         "Use `psvr_mape()` for warm-start, or `tune::tune_grid()` with ",
         "parallel cold-start for RMSPE cross-validation.",
         call. = FALSE)

  rlang::check_dots_empty()

  sym_type <- match.arg(sym_type)

  # missing() only works in the frame that owns the formal.
  if (missing(kernel))
    stop("`kernel` is required; build one with `make_kernel()`.", call. = FALSE)
  if (missing(gamma)) stop("`gamma` is required.", call. = FALSE)

  X <- as.matrix(X)
  y <- as.numeric(y)

  if (identical(sym_type, "none")) {
    .fit_rmspe(X, y, kernel = kernel, gamma = gamma,
               precondition = precondition)
  } else {
    .fit_rmspe_sym(X, y, kernel = kernel, gamma = gamma,
                   a = .sym_type_to_a(sym_type),
                   precondition = precondition)
  }
}
