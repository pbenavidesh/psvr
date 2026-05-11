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
#' @param alpha_init,alpha_star_init Optional length-`N` numeric warm-start
#'   vectors for the SMO solver (Theorem 5; `loss = "mape"` only).
#'   Projected via Algorithm 1 (new-samples-only shift + box clip) before
#'   the solve. `NULL` cold-starts. Defaults `NULL`.
#' @param warm_start_check Logical; if `TRUE`, validate post-projection
#'   feasibility of the warm-start vectors. Default `TRUE`.
#' @param new_mask Optional length-`N` logical vector flagging which samples
#'   are NEW relative to the previous fit (used by Algorithm 1 Step 2 to
#'   distribute the equality-constraint projection over new samples only).
#'   `NULL` infers "new = both alpha_init and alpha_star_init are exactly
#'   zero". Used internally by [psvr_cv()] for fold-to-fold carryover.
#' @param reg Reserved for future phases (extended Lagrangian). Must be
#'   `NULL`.
#' @param block_k4_enabled Logical; if `TRUE` (default, `loss = "mape"`
#'   only), enable the F7 block-k=4 SMO inner loop (Theorem 7 of
#'   arXiv:2605.01446 v3). Each outer iteration may pick a second
#'   working pair `(i_2, j_2)` and apply a 2-D joint update when the
#'   descent-guaranteed decoupling criterion holds. Set to `FALSE` to
#'   restore F4 (k=2 only) behaviour bit-identically. Ignored for
#'   `loss = "rmspe"`.
#' @param engine One of `"rcpp"` (default) or `"r"`. Selects the SMO
#'   backend implementation: the C++ core in `src/core_smo_solve.cpp`
#'   or the R reference implementation in `R/smo_solve.R`. Both
#'   produce bit-identical results; `"r"` is preserved as the
#'   reference for the Rcpp port and will be deprecated in v0.0.4.0
#'   and removed in v0.1.0. Ignored for `loss = "rmspe"` (LS-SVR uses
#'   `base::solve()` directly).
#' @param ... Currently unused; reserved for future extension.
#' @param alpha_couple Numeric between 0 and 1 (default `0.5`). Internal F7
#'   coupling penalty in the pair-2 WSS3 score
#'   `score = gain * (1 - alpha_couple * coupling)`. Exposed for
#'   empirical tuning; rarely needs adjustment. Ignored for
#'   `loss = "rmspe"` or `block_k4_enabled = FALSE`.
#' @param precomputed_Omega,precomputed_Omega_s INTERNAL — used by
#'   [psvr_cv()] to share a single full-dataset kernel matrix across
#'   folds. Users should not set these directly. Default `NULL`
#'   (per-fold construction). Ignored for `loss = "rmspe"`.
#'
#' @return An object of class `"psvr_fit"`, a list with components:
#'   \describe{
#'     \item{`loss`}{`"mape"` or `"rmspe"`.}
#'     \item{`sym`}{`NULL`, `+1L`, or `-1L`.}
#'     \item{`kernel`}{The kernel closure used.}
#'     \item{`alpha`}{For `loss = "mape"`, the dual variable `α` of length
#'       `N` (pre-pruning, i.e. across the full training set); for
#'       `loss = "rmspe"`, the LS-SVR `α` of length `N`.}
#'     \item{`alpha_star`}{For `loss = "mape"`, the dual variable `α*` of
#'       length `N`; `NULL` for `loss = "rmspe"`.}
#'     \item{`beta`}{For `loss = "mape"`, the pruned dual difference
#'       `β = α − α*` over the support-vector indices (length `n_sv`);
#'       `NULL` for `loss = "rmspe"`. Used by `predict()`.}
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
#'       precondition_applied, spectral)` describing the solve. The
#'       `spectral` slot is populated only for symmetric MAPE fits
#'       (`loss = "mape"`, `sym != NULL`) and reports Algorithm 2
#'       diagnostics (`mu`, `lambda_min_hat`, `lambda_max_hat`,
#'       `branch_taken`, `n_power_iterations`); `NULL` otherwise.}
#'   }
#'
#' @section Breaking change (psvr 0.0.2.9004):
#' Prior versions exposed the MAPE dual-difference `β = α − α*` under the
#' name `fit$alpha` (length `n_sv`, post-pruning). As of 0.0.2.9004, that
#' field is renamed to `fit$beta`, and `fit$alpha` now holds the true
#' length-`N` dual variable `α` (pre-pruning). The new `fit$alpha_star`
#' holds `α*` (length `N`, `NULL` for `loss = "rmspe"`). Downstream code
#' that read `fit$alpha` on a MAPE fit for prediction or diagnostics must
#' switch to `fit$beta`.
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
                 alpha_init       = NULL,
                 alpha_star_init  = NULL,
                 warm_start_check = TRUE,
                 new_mask         = NULL,
                 reg              = NULL,
                 block_k4_enabled = TRUE,
                 engine           = c("rcpp", "r"),
                 ...,
                 alpha_couple        = 0.5,
                 precomputed_Omega   = NULL,
                 precomputed_Omega_s = NULL) {
  engine <- match.arg(engine)
  # `precomputed_Omega` and `precomputed_Omega_s` are INTERNAL — populated by
  # psvr_cv() to share a single full-dataset kernel matrix across folds. Not
  # documented in @param; users should not set them. Ignored for loss =
  # "rmspe" (LS-SVR fitters do not yet accept precomputed kernels in F6).
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

  # Route to one of the four internal fitters.
  fit <- switch(paste(loss, ifelse(is.null(sym), "std", "sym"), sep = "_"),
    mape_std  = .fit_mape(X, y, kernel = kernel, C = C, eps = eps,
                          solver = solver, tol = tol,
                          alpha_init = alpha_init,
                          alpha_star_init = alpha_star_init,
                          warm_start_check = warm_start_check,
                          new_mask = new_mask,
                          precomputed_Omega = precomputed_Omega,
                          block_k4_enabled = block_k4_enabled,
                          alpha_couple = alpha_couple,
                          engine = engine),
    mape_sym  = .fit_mape_sym(X, y, kernel = kernel, C = C, eps = eps,
                              a = a, solver = solver, tol = tol,
                              alpha_init = alpha_init,
                              alpha_star_init = alpha_star_init,
                              warm_start_check = warm_start_check,
                              new_mask = new_mask,
                              precomputed_Omega_s = precomputed_Omega_s,
                              block_k4_enabled = block_k4_enabled,
                              alpha_couple = alpha_couple,
                              engine = engine),
    rmspe_std = .fit_rmspe(X, y, kernel = kernel, gamma = gamma,
                           precondition = precondition),
    rmspe_sym = .fit_rmspe_sym(X, y, kernel = kernel, gamma = gamma,
                               a = a, precondition = precondition)
  )

  is_mape <- loss == "mape"

  if (is_mape) {
    beta            <- fit$beta
    alpha           <- fit$alpha       # length N (pre-pruning)
    alpha_star      <- fit$alpha_star  # length N (pre-pruning)
    support_data    <- fit$X_sv
    support_targets <- fit$y_sv
    n_sv            <- length(fit$beta)
  } else {
    beta            <- NULL
    alpha           <- fit$alpha       # length N (LS-SVR)
    alpha_star      <- NULL
    support_data    <- fit$X_train
    support_targets <- NULL
    n_sv            <- fit$n_train
  }

  hyperparameters <- list(
    C     = if (is_mape) C     else NULL,
    eps   = if (is_mape) eps   else NULL,
    gamma = if (is_mape) NULL  else gamma,
    a     = a
  )

  solver_meta <- if (is_mape) {
    bk4 <- fit$block_k4
    list(backend                       = solver,
         iters                         = fit$iterations,
         converged                     = fit$converged,
         precondition_applied          = NA,
         # spectral diagnostics (F3, Algorithm 2) populated for symmetric
         # MAPE only; NULL for the non-symmetric path where Ωs is not built
         # and the spectral guard does not run.
         spectral                      = if (!is.null(sym)) fit$spectral else NULL,
         # F7 — block-k=4 telemetry (Theorem 7 of arXiv:2605.01446 v3).
         # Populated for solver = "smo"; NA for "osqp" (the QP backend
         # does not iterate via SMO and is unaffected by block_k4_enabled).
         joint_updates                 = if (!is.null(bk4)) bk4$joint_updates else 0L,
         k2_fallbacks                  = if (!is.null(bk4)) bk4$k2_fallbacks  else 0L,
         decoupling_rate               = if (!is.null(bk4)) bk4$decoupling_rate else NA_real_,
         early_phase_decoupling_rate   = if (!is.null(bk4)) bk4$early_phase_decoupling_rate else NA_real_,
         late_phase_decoupling_rate    = if (!is.null(bk4)) bk4$late_phase_decoupling_rate  else NA_real_)
  } else {
    list(backend                       = "linsolve",
         iters                         = 1L,
         converged                     = TRUE,
         precondition_applied          = isTRUE(fit$precondition_applied),
         spectral                      = NULL,
         joint_updates                 = 0L,
         k2_fallbacks                  = 0L,
         decoupling_rate               = NA_real_,
         early_phase_decoupling_rate   = NA_real_,
         late_phase_decoupling_rate    = NA_real_)
  }

  structure(
    list(
      loss            = loss,
      sym             = if (is.null(sym)) NULL else as.integer(sym),
      kernel          = kernel,
      alpha           = alpha,
      alpha_star      = alpha_star,
      beta            = beta,
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
