# Fit an epsilon-SVR with MAPE loss

Fits the percentage-error epsilon-SVR of the paper: Model 1 when
`sym_type = "none"`, and the symmetric-kernel Model 2 when `sym_type` is
`"even"` or `"odd"`. The dual is a quadratic program with the
sample-dependent box \\\|\beta_k\| \le 100C/y_k\\ and \\\sum_k \beta_k =
0\\, solved by the built-in SMO loop or by `osqp`.

## Usage

``` r
psvr_mape(
  X,
  y,
  sym_type = c("none", "even", "odd"),
  kernel,
  C,
  eps,
  solver = c("smo", "osqp"),
  tol = 0.001,
  max_iter = 100000L,
  alpha_init = NULL,
  alpha_star_init = NULL,
  warm_start_check = TRUE,
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

  Numeric matrix of training inputs, one observation per row (\\N \times
  p\\).

- y:

  Numeric vector of training targets, length \\N\\. Must satisfy \\y_k
  \> 0\\ for every \\k\\; percentage-error loss is undefined otherwise,
  and this is checked rather than coerced.

- sym_type:

  Symmetry type, one of `"none"` (default), `"even"` or `"odd"`. Maps
  onto the symmetry parameter \\a\\ of the paper: `"none"` fits Model 1
  and imposes no symmetry constraint; `"even"` sets \\a = +1\\,
  enforcing \\f(x) = f(-x)\\; `"odd"` sets \\a = -1\\, enforcing \\f(x)
  = -f(-x)\\. This is the same vocabulary as the `sym_type` argument of
  the parsnip specifications, so the two public surfaces agree. The
  symmetric variants require a kernel satisfying Assumption 3 of the
  paper – see
  [`make_kernel()`](https://pbenavidesh.github.io/psvr/reference/make_kernel.md).

- kernel:

  A kernel function created by
  [`make_kernel()`](https://pbenavidesh.github.io/psvr/reference/make_kernel.md).

- C:

  Regularization parameter, \\C \> 0\\. Required.

- eps:

  Insensitivity tube half-width in percentage units, \\\epsilon \ge 0\\.
  Required.

- solver:

  Backend for the dual quadratic program, `"smo"` (default) or `"osqp"`.
  See the section above.

- tol:

  Numerical tolerance, default `1e-3`. **Its meaning depends on
  `solver`.** Under `"smo"` it is the convergence tolerance of the SMO
  loop, and tighter values produce more iterations. Under `"osqp"` the
  solver runs at its own fixed tolerances and `tol` is used only
  afterwards, as the threshold below which a dual variable counts as
  zero when identifying free, saturated and support-vector sets.

- max_iter:

  Maximum SMO iterations, default `100000L`. The solver emits a
  [`warning()`](https://rdrr.io/r/base/warning.html) and returns
  `converged = FALSE` if it does not converge within `max_iter`. Ignored
  for `solver = "osqp"`.

- alpha_init, alpha_star_init:

  Optional warm-start vectors for the SMO solver, each a finite numeric
  vector of length \\N\\. Projected onto \\\sum_k (\alpha_k -
  \alpha^\*\_k) = 0\\ intersected with the per-sample box \\\[0,
  100C/y_k\]\\ before the solve. `NULL` (default) cold-starts.
  [`psvr_cv()`](https://pbenavidesh.github.io/psvr/reference/psvr_cv.md)
  manages these automatically across folds.

- warm_start_check:

  Logical; if `TRUE` (default), validate post-projection feasibility and
  [`stop()`](https://rdrr.io/r/base/stop.html) on violation. A surviving
  equality residual is fatal rather than cosmetic: SMO conserves
  \\\sum_k (\alpha_k - \alpha^\*\_k)\\, so an infeasible start is
  carried through to the returned solution.

- block_k4_enabled:

  Logical; if `TRUE` (default), enable the block-k=4 SMO inner loop.
  Each outer iteration may select a second working pair and apply a
  two-dimensional joint update when the descent-guaranteed decoupling
  criterion holds. `FALSE` restores the k=2 behaviour bit-identically.

- engine:

  One of `"rcpp"` (default) or `"r"`. Selects the SMO backend: the C++
  core, or the R reference implementation. Both produce bit-identical
  results on the development toolchain; `"r"` is retained as the
  reference and will be deprecated in 0.0.4.0 and removed in 0.1.0.

- ...:

  Must be empty. Present only so that `alpha_couple`,
  `precomputed_Omega` and `precomputed_Omega_s` must be matched by their
  exact names rather than by position or partial matching – without it
  `precomputed_Omega` would be a partial-match prefix of
  `precomputed_Omega_s`. Passing anything here is an error, which is how
  a mistyped argument name is caught.

- alpha_couple:

  Numeric in \\\[0, 1\]\\, default `0.5`. Coupling penalty in the
  second-pair selection score \\\mathrm{gain} \times (1 -
  \alpha\_{\mathrm{couple}} \cdot \mathrm{coupling})\\. Exposed for
  empirical tuning; rarely needs adjustment. Ignored when
  `block_k4_enabled = FALSE`.

- precomputed_Omega, precomputed_Omega_s:

  INTERNAL – used by
  [`psvr_cv()`](https://pbenavidesh.github.io/psvr/reference/psvr_cv.md)
  to share one full-dataset kernel matrix across folds. Users should not
  set these. `precomputed_Omega` applies when `sym_type = "none"`,
  `precomputed_Omega_s` otherwise.

## Value

For `sym_type = "none"`, an object of class `"psvr_mape"`: a list with
components `beta` (the pruned dual differences \\\beta = \alpha -
\alpha^\*\\ over the support-vector indices, used by
[`predict()`](https://rdrr.io/r/stats/predict.html)), `alpha` and
`alpha_star` (the length-\\N\\ pre-pruning duals, retained for warm
starts), `b`, `X_sv`, `y_sv`, `y_train`, `fitted_values`, `kernel`, `C`,
`eps`, `n_train`, `p_train`, `iterations`, `converged` and `block_k4`.

For `sym_type = "even"` or `"odd"`, an object of class
`"psvr_mape_sym"`: the same components plus `a` (the symmetry parameter)
and `spectral` (adaptive spectral-shift diagnostics).

Methods are available for
[`predict()`](https://rdrr.io/r/stats/predict.html),
[`print()`](https://rdrr.io/r/base/print.html),
[`coef()`](https://rdrr.io/r/stats/coef.html),
[`summary()`](https://rdrr.io/r/base/summary.html),
[`fitted()`](https://rdrr.io/r/stats/fitted.values.html) and
[`residuals()`](https://rdrr.io/r/stats/residuals.html).

## Details

For the least-squares / RMSPE family (Models 3 and 4) see
[`psvr_rmspe()`](https://pbenavidesh.github.io/psvr/reference/psvr_rmspe.md).
The two are deliberately separate functions: they share no solver, no
dual structure and no hyperparameter search space, so a single signature
would make most of its own arguments conditional. The name
[`psvr()`](https://pbenavidesh.github.io/psvr/reference/psvr-package.md)
is reserved for a future automatic-selection front end and is **not** a
synonym for either.

## Choosing a solver

`solver = "smo"` (the default) uses the built-in sequential minimal
optimisation loop. It does **not** reliably converge within `max_iter`
on linear and polynomial kernels; `converged` is `FALSE` and
`iterations` reaches the cap. The RBF kernel is unaffected. Prefer
`solver = "osqp"` for linear and polynomial kernels when accuracy
matters.

## See also

[`psvr_rmspe()`](https://pbenavidesh.github.io/psvr/reference/psvr_rmspe.md)
for the LS-SVR / RMSPE family,
[`psvr_cv()`](https://pbenavidesh.github.io/psvr/reference/psvr_cv.md)
for cross-validation with warm-start carryover,
[`make_kernel()`](https://pbenavidesh.github.io/psvr/reference/make_kernel.md)
for kernels.

## Examples

``` r
set.seed(1)
X <- matrix(rnorm(40), 20, 2)
y <- rlnorm(20)
K <- make_kernel("rbf", sigma = 1)

fit <- psvr_mape(X, y, kernel = K, C = 10, eps = 5)
predict(fit, X[1:3, , drop = FALSE])
#> [1] 0.8068941 0.7370469 0.8548128

# Even-symmetric variant (Model 2): f(x) = f(-x).
fit_sym <- psvr_mape(X, y, sym_type = "even", kernel = K, C = 10, eps = 5)
predict(fit_sym, X[1:3, , drop = FALSE])
#> [1] 0.8909758 0.7372706 0.5317936
```
