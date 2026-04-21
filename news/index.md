# Changelog

## psvr 0.0.0.9002

### User-friendly hyperparameter defaults

- New function
  [`margin_percentage()`](https://pbenavidesh.github.io/psvr/reference/margin_percentage.md):
  a dials parameter for the epsilon tube half-width in MAPE models,
  expressed in percentage units with default range `[1, 20]`. Replaces
  the absolute-unit
  [`dials::svm_margin()`](https://dials.tidymodels.org/reference/cost.html)
  for all 6 MAPE specs.

- New function
  [`sigma_heuristic()`](https://pbenavidesh.github.io/psvr/reference/sigma_heuristic.md):
  returns the median pairwise Euclidean distance of a predictor matrix —
  a standard data-driven starting point for the RBF kernel bandwidth
  (Schölkopf & Smola, 2002).

- New function
  [`rbf_sigma_psvr()`](https://pbenavidesh.github.io/psvr/reference/rbf_sigma_psvr.md):
  a dials parameter for the RBF kernel bandwidth with a `finalize` hook
  that auto-sets the search range using
  [`sigma_heuristic()`](https://pbenavidesh.github.io/psvr/reference/sigma_heuristic.md)
  when training data are available. Replaces
  [`dials::rbf_sigma()`](https://dials.tidymodels.org/reference/rbf_sigma.html)
  for all RBF specs.

- New function
  [`cost_psvr()`](https://pbenavidesh.github.io/psvr/reference/cost_psvr.md):
  a dials parameter for the regularisation parameter with range
  `[-2, 10]` on the log2 scale (approx. 0.25 to 1024), wider than
  [`dials::cost()`](https://dials.tidymodels.org/reference/cost.html) to
  accommodate the larger values typically needed by LS-SVR models.
  Replaces
  [`dials::cost()`](https://dials.tidymodels.org/reference/cost.html)
  for all 12 specs.

## psvr 0.0.0.9001

### tidymodels / parsnip integration — breaking change

Expanded parsnip integration from 4 to 12 model specs following the
tidymodels pattern used by
[`svm_rbf()`](https://parsnip.tidymodels.org/reference/svm_rbf.html),
[`svm_poly()`](https://parsnip.tidymodels.org/reference/svm_poly.html),
and
[`svm_linear()`](https://parsnip.tidymodels.org/reference/svm_linear.html):

- [`psvr_mape_rbf()`](https://pbenavidesh.github.io/psvr/reference/psvr_mape_specs.md),
  [`psvr_mape_poly()`](https://pbenavidesh.github.io/psvr/reference/psvr_mape_specs.md),
  [`psvr_mape_linear()`](https://pbenavidesh.github.io/psvr/reference/psvr_mape_specs.md)
  — Model 1 specs.
- [`psvr_mape_sym_rbf()`](https://pbenavidesh.github.io/psvr/reference/psvr_mape_sym_specs.md),
  [`psvr_mape_sym_poly()`](https://pbenavidesh.github.io/psvr/reference/psvr_mape_sym_specs.md),
  [`psvr_mape_sym_linear()`](https://pbenavidesh.github.io/psvr/reference/psvr_mape_sym_specs.md)
  — Model 2 specs.
- [`psvr_rmspe_rbf()`](https://pbenavidesh.github.io/psvr/reference/psvr_rmspe_specs.md),
  [`psvr_rmspe_poly()`](https://pbenavidesh.github.io/psvr/reference/psvr_rmspe_specs.md),
  [`psvr_rmspe_linear()`](https://pbenavidesh.github.io/psvr/reference/psvr_rmspe_specs.md)
  — Model 3 specs.
- [`psvr_rmspe_sym_rbf()`](https://pbenavidesh.github.io/psvr/reference/psvr_rmspe_sym_specs.md),
  [`psvr_rmspe_sym_poly()`](https://pbenavidesh.github.io/psvr/reference/psvr_rmspe_sym_specs.md),
  [`psvr_rmspe_sym_linear()`](https://pbenavidesh.github.io/psvr/reference/psvr_rmspe_sym_specs.md)
  — Model 4 specs.

Kernel parameters are now tunable parsnip args mapped to existing dials
params: `rbf_sigma` →
[`dials::rbf_sigma()`](https://dials.tidymodels.org/reference/rbf_sigma.html),
`degree` →
[`dials::degree()`](https://dials.tidymodels.org/reference/degree.html),
`scale_factor` →
[`dials::scale_factor()`](https://dials.tidymodels.org/reference/rbf_sigma.html).
The kernel closure is built inside each fit wrapper and no longer
appears in the parsnip layer.

The old single-spec API (`psvr_mape()`, `psvr_mape_sym()`,
`psvr_rmspe()`, `psvr_rmspe_sym()`) has been removed. Migrate by
replacing, for example,
`psvr_rmspe(cost = tune()) |> set_engine("psvr", kernel = K)` with
`psvr_rmspe_rbf(cost = tune(), rbf_sigma = 1) |> set_engine("psvr")`.

### Documentation and testing

- All vignettes and pkgdown articles updated to the new spec-based API.
- 12 new smoke tests (fit + predict) for all specs in
  `tests/testthat/test-parsnip.R`.
- pkgdown articles reorganised into three named groups: *Get Started*,
  *Case Studies*, and *Technical Notes*.

## psvr 0.0.0.9000

Initial development release.

### New models

- [`mape_svr()`](https://pbenavidesh.github.io/psvr/reference/mape_svr.md)
  — epsilon-SVR with MAPE loss (Model 1). Solves the dual QP via `osqp`
  with per-sample box constraints `|βk| ≤ 100C/yₖ`.

- [`mape_sym_svr()`](https://pbenavidesh.github.io/psvr/reference/mape_sym_svr.md)
  — symmetric epsilon-SVR with MAPE loss (Model 2). Enforces
  `f(x) = a·f(-x)` by replacing the kernel with the symmetric kernel
  `Ks(xi, xj) = K(xi, xj) + a·K(xi, -xj)`.

- [`rmspe_lssvr()`](https://pbenavidesh.github.io/psvr/reference/rmspe_lssvr.md)
  — LS-SVR with RMSPE loss (Model 3). Solves the (N+1)×(N+1) linear
  system directly via
  [`base::solve()`](https://rdrr.io/r/base/solve.html).

- [`rmspe_sym_lssvr()`](https://pbenavidesh.github.io/psvr/reference/rmspe_sym_lssvr.md)
  — symmetric LS-SVR with RMSPE loss (Model 4). Same linear system as
  Model 3 with the symmetrized kernel matrix `Ωs = ½(Ω + a·Ω*)`.

All four models are derived from a unified percentage-error loss
framework; see Benavides-Herrera et al. (2026) for the mathematical
proofs.

### Kernel interface

- [`make_kernel()`](https://pbenavidesh.github.io/psvr/reference/make_kernel.md)
  — factory returning a kernel closure from `type` ∈
  `{"rbf", "linear", "polynomial"}`. RBF and even-degree polynomial
  kernels satisfy the symmetry assumption (Assumption 3) required by
  Models 2 and 4.

### tidymodels / parsnip integration

Four parsnip model specifications for use within tidymodels workflows:

- `psvr_mape()` — spec for Model 1; hyperparameters `cost` and
  `svm_margin`.
- `psvr_mape_sym()` — spec for Model 2; hyperparameters `cost` and
  `svm_margin`.
- `psvr_rmspe()` — spec for Model 3; hyperparameter `cost` (maps to
  `Γ`).
- `psvr_rmspe_sym()` — spec for Model 4; hyperparameter `cost` (maps to
  `Γ`).

The kernel and (for symmetric models) symmetry parameter `a` are engine
arguments supplied via `set_engine("psvr", kernel = ..., a = ...)`.
Hyperparameters map to
[`dials::cost()`](https://dials.tidymodels.org/reference/cost.html) and
[`dials::svm_margin()`](https://dials.tidymodels.org/reference/cost.html)
for compatibility with
[`tune_grid()`](https://tune.tidymodels.org/reference/tune_grid.html).

### Testing

- 84 unit tests covering all four models: kernel correctness,
  QP/linear-system solutions, support vector selection, bias recovery,
  predict consistency, and input validation.
