#' Fit a percentage-error SVR / LS-SVR model
#'
#' Unified entry point for the four model families in the psvr package:
#' MAPE epsilon-SVR (Model 1), symmetric MAPE epsilon-SVR (Model 2),
#' RMSPE LS-SVR (Model 3), and symmetric RMSPE LS-SVR (Model 4). Selection
#' is driven by `loss` (`"mape"` or `"rmspe"`) and `sym` (`NULL`, `+1L`,
#' or `-1L`). The four legacy public fitters
#' ([mape_svr()], [mape_sym_svr()], [rmspe_lssvr()], [rmspe_sym_lssvr()])
#' remain available but are slated for deprecation.
#'
#' @section Cross-loss arguments:
#' Some arguments apply only to one family. When `loss = "mape"`, `gamma`
#' and `precondition` are ignored (with a warning if supplied non-`NULL`).
#' When `loss = "rmspe"`, `C`, `eps`, `solver`, and `tol` are ignored
#' (same warning rule). Default values do not trigger warnings — only
#' user-supplied values do, detected via `missing()`.
#'
#' @param X Numeric matrix of training inputs, one observation per row (N × p).
#' @param y Numeric vector of training targets (length N). Must satisfy `y > 0`.
#' @param loss One of `"mape"` (epsilon-SVR with MAPE loss) or `"rmspe"`
#'   (LS-SVR with RMSPE loss).
#' @param sym Symmetry knob. `NULL` (default) fits the non-symmetric model;
#'   `+1L` enforces even symmetry `f(x) = f(-x)`; `-1L` enforces odd symmetry
#'   `f(x) = -f(-x)`. The symmetric-kernel assumption (Assumption 3 of the
#'   paper) must hold; see [make_kernel()].
#' @param kernel A kernel function created by [make_kernel()].
#' @param C Regularization parameter `C > 0` (`loss = "mape"` only).
#' @param eps Insensitivity tube half-width `eps >= 0` in percentage units
#'   (`loss = "mape"` only).
#' @param gamma Regularization parameter `Γ > 0` (`loss = "rmspe"` only).
#' @param solver Backend for the dual QP, `"smo"` (default) or `"osqp"`
#'   (`loss = "mape"` only).
#' @param tol Solver zero-threshold (`loss = "mape"` only).
#' @param precondition One of `"auto"` (default), `"always"`, `"never"`, or
#'   a positive numeric threshold; controls Remark-17 symmetric rescaling
#'   (`loss = "rmspe"` only). See [rmspe_lssvr()] for semantics.
#' @param alpha_init,alpha_star_init,reg Reserved for future phases (warm
#'   start, extended Lagrangian). Must be `NULL` in F1.
#' @param ... Currently unused; reserved for future extension.
#'
#' @return An object of class `"psvr_fit"`, a list with components:
#'   \describe{
#'     \item{`loss`}{`"mape"` or `"rmspe"`.}
#'     \item{`sym`}{`NULL`, `+1L`, or `-1L`.}
#'     \item{`kernel`}{The kernel closure used.}
#'     \item{`alpha`}{Dual coefficients. For `loss = "mape"` this holds
#'       `α − α* = β`; for `loss = "rmspe"` it holds the LS-SVR `α`.}
#'     \item{`b`}{Bias term.}
#'     \item{`support_data`}{Support-vector matrix (after pruning) for
#'       `loss = "mape"`, or the full training matrix `X` for
#'       `loss = "rmspe"`.}
#'     \item{`support_targets`}{Support-vector targets for `loss = "mape"`;
#'       `NULL` for `loss = "rmspe"`.}
#'     \item{`n_train`, `n_sv`, `p_train`}{Training counts.}
#'     \item{`hyperparameters`}{Named list `(C, eps, gamma, a)` with `NULL`
#'       entries for the family that doesn't apply.}
#'     \item{`solver_meta`}{Named list `(backend, iters, converged,
#'       precondition_applied)` describing the solve.}
#'   }
#'
#' @examples
#' set.seed(1)
#' X <- matrix(rnorm(40), 20, 2)
#' y <- rlnorm(20)
#' K <- make_kernel("rbf", sigma = 1)
#'
#' fit_mape  <- psvr(X, y, loss = "mape",  kernel = K, C = 10, eps = 5)
#' fit_rmspe <- psvr(X, y, loss = "rmspe", kernel = K, gamma = 100)
#' fit_sym   <- psvr(X, y, loss = "rmspe", sym = +1L, kernel = K, gamma = 100)
#'
#' predict(fit_mape,  X[1:3, , drop = FALSE])
#' predict(fit_rmspe, X[1:3, , drop = FALSE])
#' predict(fit_sym,   X[1:3, , drop = FALSE])
#'
#' @export
psvr <- function(X, y,
                 loss = c("mape", "rmspe"),
                 sym  = NULL,
                 kernel,
                 C     = NULL,
                 eps   = NULL,
                 gamma = NULL,
                 solver = c("smo", "osqp"),
                 tol    = 1e-5,
                 precondition = "auto",
                 alpha_init      = NULL,
                 alpha_star_init = NULL,
                 reg             = NULL,
                 ...) {
  # missing() must be evaluated in the function's own frame; capture flags
  # here so the validator can distinguish user-passed vs. default values
  # for cross-loss warnings (match.arg defaults are otherwise opaque).
  passed <- list(
    C            = !missing(C),
    eps          = !missing(eps),
    gamma        = !missing(gamma),
    solver       = !missing(solver),
    tol          = !missing(tol),
    precondition = !missing(precondition)
  )

  loss   <- match.arg(loss)
  solver <- match.arg(solver)

  X <- as.matrix(X)
  y <- as.numeric(y)

  a <- if (is.null(sym)) NULL else as.integer(sym)

  .validate_psvr_inputs(X = X, y = y, loss = loss, sym = sym,
                        C = C, eps = eps, gamma = gamma, a = a,
                        alpha_init = alpha_init,
                        alpha_star_init = alpha_star_init,
                        reg = reg,
                        passed = passed)

  # Route to one of the four fitters. In Step 4 these are still under
  # their public names; Step 5 will rename them to `.fit_*` and update
  # the routing in lock-step.
  fit <- switch(paste(loss, ifelse(is.null(sym), "std", "sym"), sep = "_"),
    mape_std  = mape_svr(X, y, kernel = kernel, C = C, eps = eps,
                         solver = solver, tol = tol),
    mape_sym  = mape_sym_svr(X, y, kernel = kernel, C = C, eps = eps,
                             a = a, solver = solver, tol = tol),
    rmspe_std = rmspe_lssvr(X, y, kernel = kernel, gamma = gamma,
                            precondition = precondition),
    rmspe_sym = rmspe_sym_lssvr(X, y, kernel = kernel, gamma = gamma,
                                a = a, precondition = precondition)
  )

  is_mape <- loss == "mape"

  alpha           <- if (is_mape) fit$beta    else fit$alpha
  support_data    <- if (is_mape) fit$X_sv    else fit$X_train
  support_targets <- if (is_mape) fit$y_sv    else NULL
  n_sv            <- if (is_mape) length(fit$beta) else fit$n_train

  hyperparameters <- list(
    C     = if (is_mape) C     else NULL,
    eps   = if (is_mape) eps   else NULL,
    gamma = if (is_mape) NULL  else gamma,
    a     = a
  )

  solver_meta <- if (is_mape) {
    list(backend              = solver,
         iters                = NA_integer_,
         converged            = NA,
         precondition_applied = NA)
  } else {
    list(backend              = "linsolve",
         iters                = 1L,
         converged            = TRUE,
         precondition_applied = isTRUE(fit$precondition_applied))
  }

  structure(
    list(
      loss            = loss,
      sym             = if (is.null(sym)) NULL else as.integer(sym),
      kernel          = kernel,
      alpha           = alpha,
      b               = fit$b,
      support_data    = support_data,
      support_targets = support_targets,
      n_train         = fit$n_train,
      n_sv            = n_sv,
      p_train         = fit$p_train,
      hyperparameters = hyperparameters,
      solver_meta     = solver_meta
    ),
    class = "psvr_fit"
  )
}
