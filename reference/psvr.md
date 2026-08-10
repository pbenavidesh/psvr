# Fit a percentage-error SVR / LS-SVR model

Unified entry point for the four model families in the psvr package:
MAPE epsilon-SVR (Model 1), symmetric MAPE epsilon-SVR (Model 2), RMSPE
LS-SVR (Model 3), and symmetric RMSPE LS-SVR (Model 4). Selection is
driven by `loss` (`"mape"` or `"rmspe"`) and `sym` (`NULL`, `+1L`, or
`-1L`). The four legacy public fitters
([`mape_svr()`](https://pbenavidesh.github.io/psvr/reference/mape_svr.md),
[`mape_sym_svr()`](https://pbenavidesh.github.io/psvr/reference/mape_sym_svr.md),
[`rmspe_lssvr()`](https://pbenavidesh.github.io/psvr/reference/rmspe_lssvr.md),
[`rmspe_sym_lssvr()`](https://pbenavidesh.github.io/psvr/reference/rmspe_sym_lssvr.md))
remain available but are slated for deprecation.

## Usage

``` r
psvr(
  X,
  y,
  loss = c("mape", "rmspe"),
  sym = NULL,
  kernel,
  C = NULL,
  eps = NULL,
  gamma = NULL,
  solver = c("smo", "osqp"),
  tol = 0.001,
  max_iter = 100000L,
  precondition = "auto",
  alpha_init = NULL,
  alpha_star_init = NULL,
  warm_start_check = TRUE,
  reg = NULL,
  block_k4_enabled = TRUE,
  engine = c("rcpp", "r"),
  ...,
  alpha_couple = 0.5,
  precomputed_Omega = NULL,
  precomputed_Omega_s = NULL
)
```

## Arguments

- X:

  Numeric matrix of training inputs, one observation per row (N × p).

- y:

  Numeric vector of training targets (length N). Must satisfy `y > 0`.

- loss:

  One of `"mape"` (epsilon-SVR with MAPE loss) or `"rmspe"` (LS-SVR with
  RMSPE loss).

- sym:

  Symmetry knob. `NULL` (default) fits the non-symmetric model; `+1L`
  enforces even symmetry `f(x) = f(-x)`; `-1L` enforces odd symmetry
  `f(x) = -f(-x)`. The symmetric-kernel assumption (Assumption 3 of the
  paper) must hold; see
  [`make_kernel()`](https://pbenavidesh.github.io/psvr/reference/make_kernel.md).

- kernel:

  A kernel function created by
  [`make_kernel()`](https://pbenavidesh.github.io/psvr/reference/make_kernel.md).

- C:

  Regularization parameter `C > 0` (`loss = "mape"` only).

- eps:

  Insensitivity tube half-width `eps >= 0` in percentage units
  (`loss = "mape"` only).

- gamma:

  Regularization parameter `Γ > 0` (`loss = "rmspe"` only).

- solver:

  Backend for the dual QP, `"smo"` (default) or `"osqp"`
  (`loss = "mape"` only).

- tol:

  Solver convergence tolerance for the SMO loop (`loss = "mape"` only).
  Default `1e-3`. Tighter values produce more iterations.

- max_iter:

  Maximum SMO iterations (`loss = "mape"` only). Default `100000L`. The
  solver emits a [`warning()`](https://rdrr.io/r/base/warning.html) and
  returns `solver_meta$converged = FALSE` if it does not converge within
  `max_iter`.

- precondition:

  One of `"auto"` (default), `"always"`, `"never"`, or a positive
  numeric threshold; controls Remark-17 symmetric rescaling
  (`loss = "rmspe"` only). See
  [`rmspe_lssvr()`](https://pbenavidesh.github.io/psvr/reference/rmspe_lssvr.md)
  for semantics.

- alpha_init, alpha_star_init:

  Optional length-`N` numeric warm-start vectors for the SMO solver
  (Theorem 5; `loss = "mape"` only). Projected via Algorithm 1 — the
  exact minimum-norm (Euclidean) projection onto
  `sum(alpha - alpha_star) = 0` intersected with the per-sample box
  `[0, 100C/y_k]` — before the solve. `NULL` cold-starts. Defaults
  `NULL`.

- warm_start_check:

  Logical; if `TRUE` (default), validate post-projection feasibility of
  the warm-start vectors and
  [`stop()`](https://rdrr.io/r/base/stop.html) on violation. A surviving
  equality residual is fatal rather than cosmetic: SMO conserves
  `sum(alpha - alpha_star)`, so an infeasible start is carried through
  to the returned solution. Set `FALSE` to override.

- reg:

  Reserved for future phases (extended Lagrangian). Must be `NULL`.

- block_k4_enabled:

  Logical; if `TRUE` (default, `loss = "mape"` only), enable the F7
  block-k=4 SMO inner loop (Theorem 7 of arXiv:2605.01446 v3). Each
  outer iteration may pick a second working pair `(i_2, j_2)` and apply
  a 2-D joint update when the descent-guaranteed decoupling criterion
  holds. Set to `FALSE` to restore F4 (k=2 only) behaviour
  bit-identically. Ignored for `loss = "rmspe"`.

- engine:

  One of `"rcpp"` (default) or `"r"`. Selects the SMO backend
  implementation: the C++ core in `src/core_smo_solve.cpp` or the R
  reference implementation in `R/smo_solve.R`. Both produce
  bit-identical results; `"r"` is preserved as the reference for the
  Rcpp port and will be deprecated in v0.0.4.0 and removed in v0.1.0.
  Ignored for `loss = "rmspe"` (LS-SVR uses
  [`base::solve()`](https://rdrr.io/r/base/solve.html) directly).

- ...:

  Must be empty. Present only so that `alpha_couple`,
  `precomputed_Omega`, and `precomputed_Omega_s` must be matched by
  their exact names rather than by position or partial matching. Passing
  anything here is an error, which is how a mistyped argument name is
  caught.

- alpha_couple:

  Numeric between 0 and 1 (default `0.5`). Internal F7 coupling penalty
  in the pair-2 WSS3 score
  `score = gain * (1 - alpha_couple * coupling)`. Exposed for empirical
  tuning; rarely needs adjustment. Ignored for `loss = "rmspe"` or
  `block_k4_enabled = FALSE`.

- precomputed_Omega, precomputed_Omega_s:

  INTERNAL — used by
  [`psvr_cv()`](https://pbenavidesh.github.io/psvr/reference/psvr_cv.md)
  to share a single full-dataset kernel matrix across folds. Users
  should not set these directly. Default `NULL` (per-fold construction).
  Ignored for `loss = "rmspe"`.

## Value

An object of class `"psvr_fit"`, a list with components:

- `loss`:

  `"mape"` or `"rmspe"`.

- `sym`:

  `NULL`, `+1L`, or `-1L`.

- `kernel`:

  The kernel closure used.

- `alpha`:

  For `loss = "mape"`, the dual variable `α` of length `N` (pre-pruning,
  i.e. across the full training set); for `loss = "rmspe"`, the LS-SVR
  `α` of length `N`.

- `alpha_star`:

  For `loss = "mape"`, the dual variable `α*` of length `N`; `NULL` for
  `loss = "rmspe"`.

- `beta`:

  For `loss = "mape"`, the pruned dual difference `β = α − α*` over the
  support-vector indices (length `n_sv`); `NULL` for `loss = "rmspe"`.
  Used by [`predict()`](https://rdrr.io/r/stats/predict.html).

- `b`:

  Bias term.

- `support_data`:

  Support-vector matrix (after pruning) for `loss = "mape"`, or the full
  training matrix `X` for `loss = "rmspe"`.

- `support_targets`:

  Support-vector targets for `loss = "mape"`; `NULL` for
  `loss = "rmspe"`.

- `y_train`:

  The length-`N` training targets, retained so that
  [`fitted.psvr_fit()`](https://pbenavidesh.github.io/psvr/reference/fitted.psvr_fit.md)
  and
  [`residuals.psvr_fit()`](https://pbenavidesh.github.io/psvr/reference/residuals.psvr_fit.md)
  need no refit.

- `fitted_values`:

  The length-`N` in-sample predictions `f(x_k)`, computed during the
  fit. Equal to `predict(object, X_train)` to machine precision, but
  recovered from state the solver already holds rather than by
  rebuilding the `N × N` kernel matrix.

- `n_train`, `n_sv`, `p_train`:

  Training counts.

- `hyperparameters`:

  Named list `(C, eps, gamma, a)` with `NULL` entries for the family
  that doesn't apply.

- `solver_meta`:

  Named list
  `(backend, iters, converged, precondition_applied, spectral)`
  describing the solve. The `spectral` slot is populated only for
  symmetric MAPE fits (`loss = "mape"`, `sym != NULL`) and reports
  Algorithm 2 diagnostics (`mu`, `lambda_min_hat`, `lambda_max_hat`,
  `branch_taken`, `n_power_iterations`); `NULL` otherwise.

## Cross-loss arguments

Some arguments apply only to one family. When `loss = "mape"`, `gamma`
and `precondition` are ignored (with a warning if supplied non-`NULL`).
When `loss = "rmspe"`, `C`, `eps`, `solver`, `tol`, and `max_iter` are
ignored (same warning rule). Default values do not trigger warnings —
only user-supplied values do, detected via
[`missing()`](https://rdrr.io/r/base/missing.html).

## Breaking change (psvr 0.0.2.9004)

Prior versions exposed the MAPE dual-difference `β = α − α*` under the
name `fit$alpha` (length `n_sv`, post-pruning). As of 0.0.2.9004, that
field is renamed to `fit$beta`, and `fit$alpha` now holds the true
length-`N` dual variable `α` (pre-pruning). The new `fit$alpha_star`
holds `α*` (length `N`, `NULL` for `loss = "rmspe"`). Downstream code
that read `fit$alpha` on a MAPE fit for prediction or diagnostics must
switch to `fit$beta`.

## Examples

``` r
set.seed(1)
X <- matrix(rnorm(40), 20, 2)
y <- rlnorm(20)
K <- make_kernel("rbf", sigma = 1)

fit_mape  <- psvr(X, y, loss = "mape",  kernel = K, C = 10, eps = 5)
fit_rmspe <- psvr(X, y, loss = "rmspe", kernel = K, gamma = 100)
fit_sym   <- psvr(X, y, loss = "rmspe", sym = +1L, kernel = K, gamma = 100)

predict(fit_mape,  X[1:3, , drop = FALSE])
#> [1] 0.8068941 0.7370469 0.8548128
predict(fit_rmspe, X[1:3, , drop = FALSE])
#> [1] 0.9189556 0.7369731 0.8524386
predict(fit_sym,   X[1:3, , drop = FALSE])
#> [1] 1.1583805 0.7439030 0.5084087
```
