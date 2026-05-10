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
  tol = 1e-05,
  precondition = "auto",
  alpha_init = NULL,
  alpha_star_init = NULL,
  reg = NULL,
  ...
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

  Solver zero-threshold (`loss = "mape"` only).

- precondition:

  One of `"auto"` (default), `"always"`, `"never"`, or a positive
  numeric threshold; controls Remark-17 symmetric rescaling
  (`loss = "rmspe"` only). See
  [`rmspe_lssvr()`](https://pbenavidesh.github.io/psvr/reference/rmspe_lssvr.md)
  for semantics.

- alpha_init, alpha_star_init, reg:

  Reserved for future phases (warm start, extended Lagrangian). Must be
  `NULL` in F1.

- ...:

  Currently unused; reserved for future extension.

## Value

An object of class `"psvr_fit"`, a list with components:

- `loss`:

  `"mape"` or `"rmspe"`.

- `sym`:

  `NULL`, `+1L`, or `-1L`.

- `kernel`:

  The kernel closure used.

- `alpha`:

  Dual coefficients. For `loss = "mape"` this holds `α − α* = β`; for
  `loss = "rmspe"` it holds the LS-SVR `α`.

- `b`:

  Bias term.

- `support_data`:

  Support-vector matrix (after pruning) for `loss = "mape"`, or the full
  training matrix `X` for `loss = "rmspe"`.

- `support_targets`:

  Support-vector targets for `loss = "mape"`; `NULL` for
  `loss = "rmspe"`.

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
When `loss = "rmspe"`, `C`, `eps`, `solver`, and `tol` are ignored (same
warning rule). Default values do not trigger warnings — only
user-supplied values do, detected via
[`missing()`](https://rdrr.io/r/base/missing.html).

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
#> [1] 0.8066870 0.7365194 0.8556591
predict(fit_rmspe, X[1:3, , drop = FALSE])
#> [1] 0.9189556 0.7369731 0.8524386
predict(fit_sym,   X[1:3, , drop = FALSE])
#> [1] 1.1583805 0.7439030 0.5084087
```
