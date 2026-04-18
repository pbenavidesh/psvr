# Changelog

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

- [`psvr_mape()`](https://pbenavidesh.github.io/psvr/reference/psvr_mape.md)
  — spec for Model 1; hyperparameters `cost` and `svm_margin`.
- [`psvr_mape_sym()`](https://pbenavidesh.github.io/psvr/reference/psvr_mape_sym.md)
  — spec for Model 2; hyperparameters `cost` and `svm_margin`.
- [`psvr_rmspe()`](https://pbenavidesh.github.io/psvr/reference/psvr_rmspe.md)
  — spec for Model 3; hyperparameter `cost` (maps to `Γ`).
- [`psvr_rmspe_sym()`](https://pbenavidesh.github.io/psvr/reference/psvr_rmspe_sym.md)
  — spec for Model 4; hyperparameter `cost` (maps to `Γ`).

The kernel and (for symmetric models) symmetry parameter `a` are engine
arguments supplied via `set_engine("psvr", kernel = ..., a = ...)`.
Hyperparameters map to `dials::cost()` and `dials::svm_margin()` for
compatibility with `tune_grid()`.

### Testing

- 84 unit tests covering all four models: kernel correctness,
  QP/linear-system solutions, support vector selection, bias recovery,
  predict consistency, and input validation.
