# Changelog

## psvr 0.0.2 (unreleased)

### New features

- The `precondition` argument from
  [`rmspe_lssvr()`](https://pbenavidesh.github.io/psvr/reference/rmspe_lssvr.md)
  and
  [`rmspe_sym_lssvr()`](https://pbenavidesh.github.io/psvr/reference/rmspe_sym_lssvr.md)
  is now configurable via parsnip’s
  [`set_engine()`](https://parsnip.tidymodels.org/reference/set_engine.html)
  for all six RMSPE spec functions (rbf/poly/linear, both symmetric and
  non-symmetric). Default is `"auto"`; pass via
  `set_engine("psvr", precondition = "always")`, `"never"`, or a numeric
  threshold. Not registered as a tunable parameter — it is a
  configuration flag, not a hyperparameter.

## psvr 0.0.1

### New features

- Added optional preconditioner for LS-SVR variants
  ([`rmspe_lssvr()`](https://pbenavidesh.github.io/psvr/reference/rmspe_lssvr.md),
  [`rmspe_sym_lssvr()`](https://pbenavidesh.github.io/psvr/reference/rmspe_sym_lssvr.md))
  via the `precondition` argument. Accepted values: “always”, “never”,
  “auto” (default; activates when max(y)/min(y) \> 10), or a numeric
  threshold. The preconditioner is a mathematically exact change of
  variable that improves numerical conditioning at large target dynamic
  ranges without changing predictions in exact arithmetic.
- Returned model objects now include `precondition_applied` (logical)
  for diagnostic transparency. The `print` method displays this field
  when TRUE.

### Documentation

- Updated `@details` for both LS-SVR fitters with the bordered-system
  formulation and the recovery formula α_k = α̃\_k / y_k.

## psvr 0.0.0.9005

#### New features

- `cost_psvr_ls_data(y, n, width_log2)` — a data-driven cost range for
  LS-SVR psvr models (`m3`, `m4`). Default upper bound scales as
  `log2(var(y) * N) + width_log2` (default `width_log2 = 4`), the
  standard heuristic for the LS-SVR regularisation parameter `Γ`
  (Suykens et al. 2002, *Least Squares Support Vector Machines*,
  §3.1.3). Use this instead of
  [`cost_psvr()`](https://pbenavidesh.github.io/psvr/reference/cost_psvr.md)
  for `m3`/`m4` workflows; the static `[-2, 10]` log2 range
  underestimates `Γ` on benchmark datasets (Boston Housing optimum
  `Γ ≈ 1.7 × 10⁴`,
  vs. [`cost_psvr()`](https://pbenavidesh.github.io/psvr/reference/cost_psvr.md)
  upper bound `2^10 = 1024`).
- `psvr_option_add_cost_ls(wf_set, y, ...)` — a convenience wrapper that
  applies
  [`cost_psvr_ls_data()`](https://pbenavidesh.github.io/psvr/reference/cost_psvr_ls_data.md)
  to every LS-SVR psvr workflow in a workflow set (those whose
  `wflow_id` matches `m3|m4`), analogous to
  [`psvr_option_add()`](https://pbenavidesh.github.io/psvr/reference/psvr_option_add.md)
  for `rbf_sigma`.

## psvr 0.0.0.9004

#### New features

- `sym_type` is now a tunable parsnip argument in
  [`psvr_mape_sym_rbf()`](https://pbenavidesh.github.io/psvr/reference/psvr_mape_sym_specs.md)
  and
  [`psvr_rmspe_sym_rbf()`](https://pbenavidesh.github.io/psvr/reference/psvr_rmspe_sym_specs.md).
  Pass `sym_type = tune()` to let CV select between even (a = 1) and odd
  (a = -1) symmetry automatically.
- Engine default `a = 1L` removed from symmetric RBF specs; symmetry
  type is now controlled exclusively through the `sym_type` model
  argument.

## psvr 0.0.0.9003

### UX improvements

- New function
  [`rbf_sigma_psvr_data()`](https://pbenavidesh.github.io/psvr/reference/rbf_sigma_psvr_data.md):
  combines
  [`sigma_heuristic()`](https://pbenavidesh.github.io/psvr/reference/sigma_heuristic.md)
  and
  [`rbf_sigma_psvr()`](https://pbenavidesh.github.io/psvr/reference/rbf_sigma_psvr.md)
  in a single call, returning a dials parameter with a data-driven
  search range centred on the median pairwise distance.

- New function
  [`psvr_option_add()`](https://pbenavidesh.github.io/psvr/reference/psvr_option_add.md):
  applies `option_add()` to every psvr workflow (those whose `wflow_id`
  matches `m1|m2|m3|m4`) in a workflow set simultaneously, replacing the
  `rbf_sigma` parameter with a data-driven one. Replaces four individual
  `option_add()` calls in typical workflows.

- Symmetric model specs (`psvr_mape_sym_*`, `psvr_rmspe_sym_*`): even
  symmetry (`a = 1L`) is now the default engine argument. Calling
  `set_engine("psvr")` without `a = 1L` now produces the same result as
  before — no action required for existing code that already passed
  `a = 1L`.

- All four core fitting functions now emit a warning when `N > 2000` to
  remind users that the O(n²) kernel matrix may be expensive.

- The `y > 0` input check in all four core fitting functions now reports
  the number of offending values and their minimum, making it easier to
  diagnose data issues.

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

- [`rbf_sigma_psvr()`](https://pbenavidesh.github.io/psvr/reference/rbf_sigma_psvr.md):
  removed non-functional `finalize` hook. The search range must be set
  manually using
  [`sigma_heuristic()`](https://pbenavidesh.github.io/psvr/reference/sigma_heuristic.md)
  and `option_add()`. See
  [`?rbf_sigma_psvr`](https://pbenavidesh.github.io/psvr/reference/rbf_sigma_psvr.md)
  for the recommended workflow.

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
