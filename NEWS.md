# psvr 0.0.0.9001

## tidymodels / parsnip integration — breaking change

Expanded parsnip integration from 4 to 12 model specs following the tidymodels
pattern used by `svm_rbf()`, `svm_poly()`, and `svm_linear()`:

* `psvr_mape_rbf()`, `psvr_mape_poly()`, `psvr_mape_linear()` — Model 1 specs.
* `psvr_mape_sym_rbf()`, `psvr_mape_sym_poly()`, `psvr_mape_sym_linear()` — Model 2 specs.
* `psvr_rmspe_rbf()`, `psvr_rmspe_poly()`, `psvr_rmspe_linear()` — Model 3 specs.
* `psvr_rmspe_sym_rbf()`, `psvr_rmspe_sym_poly()`, `psvr_rmspe_sym_linear()` — Model 4 specs.

Kernel parameters are now tunable parsnip args mapped to existing dials params:
`rbf_sigma` → `dials::rbf_sigma()`, `degree` → `dials::degree()`,
`scale_factor` → `dials::scale_factor()`.  The kernel closure is built inside
each fit wrapper and no longer appears in the parsnip layer.

The old single-spec API (`psvr_mape()`, `psvr_mape_sym()`, `psvr_rmspe()`,
`psvr_rmspe_sym()`) has been removed.  Migrate by replacing, for example,
`psvr_rmspe(cost = tune()) |> set_engine("psvr", kernel = K)` with
`psvr_rmspe_rbf(cost = tune(), rbf_sigma = 1) |> set_engine("psvr")`.

## Documentation and testing

* All vignettes and pkgdown articles updated to the new spec-based API.
* 12 new smoke tests (fit + predict) for all specs in
  `tests/testthat/test-parsnip.R`.
* pkgdown articles reorganised into three named groups: *Get Started*,
  *Case Studies*, and *Technical Notes*.

# psvr 0.0.0.9000

Initial development release.

## New models

* `mape_svr()` — epsilon-SVR with MAPE loss (Model 1). Solves the dual QP via
  `osqp` with per-sample box constraints `|βk| ≤ 100C/yₖ`.

* `mape_sym_svr()` — symmetric epsilon-SVR with MAPE loss (Model 2). Enforces
  `f(x) = a·f(-x)` by replacing the kernel with the symmetric kernel
  `Ks(xi, xj) = K(xi, xj) + a·K(xi, -xj)`.

* `rmspe_lssvr()` — LS-SVR with RMSPE loss (Model 3). Solves the (N+1)×(N+1)
  linear system directly via `base::solve()`.

* `rmspe_sym_lssvr()` — symmetric LS-SVR with RMSPE loss (Model 4). Same
  linear system as Model 3 with the symmetrized kernel matrix
  `Ωs = ½(Ω + a·Ω*)`.

All four models are derived from a unified percentage-error loss framework;
see Benavides-Herrera et al. (2026) for the mathematical proofs.

## Kernel interface

* `make_kernel()` — factory returning a kernel closure from `type` ∈
  `{"rbf", "linear", "polynomial"}`. RBF and even-degree polynomial kernels
  satisfy the symmetry assumption (Assumption 3) required by Models 2 and 4.

## tidymodels / parsnip integration

Four parsnip model specifications for use within tidymodels workflows:

* `psvr_mape()` — spec for Model 1; hyperparameters `cost` and `svm_margin`.
* `psvr_mape_sym()` — spec for Model 2; hyperparameters `cost` and `svm_margin`.
* `psvr_rmspe()` — spec for Model 3; hyperparameter `cost` (maps to `Γ`).
* `psvr_rmspe_sym()` — spec for Model 4; hyperparameter `cost` (maps to `Γ`).

The kernel and (for symmetric models) symmetry parameter `a` are engine
arguments supplied via `set_engine("psvr", kernel = ..., a = ...)`.
Hyperparameters map to `dials::cost()` and `dials::svm_margin()` for
compatibility with `tune_grid()`.

## Testing

* 84 unit tests covering all four models: kernel correctness, QP/linear-system
  solutions, support vector selection, bias recovery, predict consistency,
  and input validation.
