# Package index

## Kernel factory

Shared kernel interface used by all four models.

- [`make_kernel()`](https://pbenavidesh.github.io/psvr/reference/make_kernel.md)
  : Create a kernel function

## Model 1 — ε-SVR with MAPE

Epsilon-insensitive SVR minimising the MAPE loss (Theorem 1).

- [`mape_svr()`](https://pbenavidesh.github.io/psvr/reference/mape_svr.md)
  : Fit epsilon-SVR with MAPE loss (Model 1)
- [`predict(`*`<psvr_mape>`*`)`](https://pbenavidesh.github.io/psvr/reference/predict.psvr_mape.md)
  : Predict from a fitted epsilon-SVR with MAPE model

## Model 2 — Symmetric ε-SVR with MAPE

Adds an even/odd symmetry constraint via the symmetric kernel
`Ks(xi, xj) = K(xi, xj) + a·K(xi, -xj)` (Theorem 2).

- [`mape_sym_svr()`](https://pbenavidesh.github.io/psvr/reference/mape_sym_svr.md)
  : Fit symmetric epsilon-SVR with MAPE loss (Model 2)
- [`predict(`*`<psvr_mape_sym>`*`)`](https://pbenavidesh.github.io/psvr/reference/predict.psvr_mape_sym.md)
  : Predict from a fitted symmetric epsilon-SVR with MAPE model

## Model 3 — LS-SVR with RMSPE

Least-squares SVR minimising the RMSPE loss (Theorem 3).

- [`rmspe_lssvr()`](https://pbenavidesh.github.io/psvr/reference/rmspe_lssvr.md)
  : Fit LS-SVR with RMSPE loss (Model 3)
- [`predict(`*`<psvr_rmspe>`*`)`](https://pbenavidesh.github.io/psvr/reference/predict.psvr_rmspe.md)
  : Predict from a fitted LS-SVR with RMSPE model

## Model 4 — Symmetric LS-SVR with RMSPE

Symmetric kernel extension of Model 3 (Theorem 4).

- [`rmspe_sym_lssvr()`](https://pbenavidesh.github.io/psvr/reference/rmspe_sym_lssvr.md)
  : Fit symmetric LS-SVR with RMSPE loss (Model 4)
- [`predict(`*`<psvr_rmspe_sym>`*`)`](https://pbenavidesh.github.io/psvr/reference/predict.psvr_rmspe_sym.md)
  : Predict from a fitted symmetric LS-SVR with RMSPE model

## Model methods

S3 methods for inspecting and printing fitted psvr models.

- [`print(`*`<psvr_mape>`*`)`](https://pbenavidesh.github.io/psvr/reference/print.psvr_mape.md)
  : Print method for psvr_mape objects
- [`print(`*`<psvr_mape_sym>`*`)`](https://pbenavidesh.github.io/psvr/reference/print.psvr_mape_sym.md)
  : Print method for psvr_mape_sym objects
- [`print(`*`<psvr_rmspe>`*`)`](https://pbenavidesh.github.io/psvr/reference/print.psvr_rmspe.md)
  : Print method for psvr_rmspe objects
- [`print(`*`<psvr_rmspe_sym>`*`)`](https://pbenavidesh.github.io/psvr/reference/print.psvr_rmspe_sym.md)
  : Print method for psvr_rmspe_sym objects
- [`coef(`*`<psvr_mape>`*`)`](https://pbenavidesh.github.io/psvr/reference/coef.psvr_mape.md)
  : Extract coefficients from a psvr_mape model
- [`coef(`*`<psvr_mape_sym>`*`)`](https://pbenavidesh.github.io/psvr/reference/coef.psvr_mape_sym.md)
  : Extract coefficients from a psvr_mape_sym model
- [`coef(`*`<psvr_rmspe>`*`)`](https://pbenavidesh.github.io/psvr/reference/coef.psvr_rmspe.md)
  : Extract coefficients from a psvr_rmspe model
- [`coef(`*`<psvr_rmspe_sym>`*`)`](https://pbenavidesh.github.io/psvr/reference/coef.psvr_rmspe_sym.md)
  : Extract coefficients from a psvr_rmspe_sym model

## tidymodels / parsnip interface — ε-SVR with MAPE (Model 1)

Parsnip model specifications for Model 1 (one per kernel type).

- [`psvr_mape_rbf()`](https://pbenavidesh.github.io/psvr/reference/psvr_mape_specs.md)
  [`psvr_mape_poly()`](https://pbenavidesh.github.io/psvr/reference/psvr_mape_specs.md)
  [`psvr_mape_linear()`](https://pbenavidesh.github.io/psvr/reference/psvr_mape_specs.md)
  : Parsnip model specs: epsilon-SVR with MAPE loss (Model 1)

## tidymodels / parsnip interface — Symmetric ε-SVR with MAPE (Model 2)

Parsnip model specifications for Model 2 (one per kernel type).

- [`psvr_mape_sym_rbf()`](https://pbenavidesh.github.io/psvr/reference/psvr_mape_sym_specs.md)
  [`psvr_mape_sym_poly()`](https://pbenavidesh.github.io/psvr/reference/psvr_mape_sym_specs.md)
  [`psvr_mape_sym_linear()`](https://pbenavidesh.github.io/psvr/reference/psvr_mape_sym_specs.md)
  : Parsnip model specs: symmetric epsilon-SVR with MAPE loss (Model 2)

## tidymodels / parsnip interface — LS-SVR with RMSPE (Model 3)

Parsnip model specifications for Model 3 (one per kernel type).

- [`psvr_rmspe_rbf()`](https://pbenavidesh.github.io/psvr/reference/psvr_rmspe_specs.md)
  [`psvr_rmspe_poly()`](https://pbenavidesh.github.io/psvr/reference/psvr_rmspe_specs.md)
  [`psvr_rmspe_linear()`](https://pbenavidesh.github.io/psvr/reference/psvr_rmspe_specs.md)
  : Parsnip model specs: LS-SVR with RMSPE loss (Model 3)

## tidymodels / parsnip interface — Symmetric LS-SVR with RMSPE (Model 4)

Parsnip model specifications for Model 4 (one per kernel type).

- [`psvr_rmspe_sym_rbf()`](https://pbenavidesh.github.io/psvr/reference/psvr_rmspe_sym_specs.md)
  [`psvr_rmspe_sym_poly()`](https://pbenavidesh.github.io/psvr/reference/psvr_rmspe_sym_specs.md)
  [`psvr_rmspe_sym_linear()`](https://pbenavidesh.github.io/psvr/reference/psvr_rmspe_sym_specs.md)
  : Parsnip model specs: symmetric LS-SVR with RMSPE loss (Model 4)

## Hyperparameter utilities

Custom dials parameters and tuning helpers for psvr models.

- [`margin_percentage()`](https://pbenavidesh.github.io/psvr/reference/margin_percentage.md)
  : Insensitivity margin in percentage units
- [`sigma_heuristic()`](https://pbenavidesh.github.io/psvr/reference/sigma_heuristic.md)
  : Median-distance heuristic for RBF kernel bandwidth
- [`rbf_sigma_psvr()`](https://pbenavidesh.github.io/psvr/reference/rbf_sigma_psvr.md)
  : RBF sigma parameter for psvr models
- [`rbf_sigma_psvr_data()`](https://pbenavidesh.github.io/psvr/reference/rbf_sigma_psvr_data.md)
  : RBF sigma parameter with data-driven range for psvr models
- [`psvr_option_add()`](https://pbenavidesh.github.io/psvr/reference/psvr_option_add.md)
  : Apply data-driven rbf_sigma to all psvr workflows in a workflow set
- [`psvr_option_add_cost_ls()`](https://pbenavidesh.github.io/psvr/reference/psvr_option_add_cost_ls.md)
  : Apply data-driven LS-SVR cost range to all m3/m4 workflows in a
  workflow set
- [`cost_psvr()`](https://pbenavidesh.github.io/psvr/reference/cost_psvr.md)
  : Cost parameter with extended range for psvr models
- [`cost_psvr_ls_data()`](https://pbenavidesh.github.io/psvr/reference/cost_psvr_ls_data.md)
  : Data-driven cost range for LS-SVR psvr models
- [`sym_type_param()`](https://pbenavidesh.github.io/psvr/reference/sym_type_param.md)
  : Dials parameter for symmetry type

## Internal fit wrappers

Low-level bridge functions called by parsnip; not for direct use.

- [`psvr_mape_rbf_fit()`](https://pbenavidesh.github.io/psvr/reference/psvr-fit-wrappers.md)
  [`psvr_mape_poly_fit()`](https://pbenavidesh.github.io/psvr/reference/psvr-fit-wrappers.md)
  [`psvr_mape_linear_fit()`](https://pbenavidesh.github.io/psvr/reference/psvr-fit-wrappers.md)
  [`psvr_mape_sym_rbf_fit()`](https://pbenavidesh.github.io/psvr/reference/psvr-fit-wrappers.md)
  [`psvr_mape_sym_poly_fit()`](https://pbenavidesh.github.io/psvr/reference/psvr-fit-wrappers.md)
  [`psvr_mape_sym_linear_fit()`](https://pbenavidesh.github.io/psvr/reference/psvr-fit-wrappers.md)
  [`psvr_rmspe_rbf_fit()`](https://pbenavidesh.github.io/psvr/reference/psvr-fit-wrappers.md)
  [`psvr_rmspe_poly_fit()`](https://pbenavidesh.github.io/psvr/reference/psvr-fit-wrappers.md)
  [`psvr_rmspe_linear_fit()`](https://pbenavidesh.github.io/psvr/reference/psvr-fit-wrappers.md)
  [`psvr_rmspe_sym_rbf_fit()`](https://pbenavidesh.github.io/psvr/reference/psvr-fit-wrappers.md)
  [`psvr_rmspe_sym_poly_fit()`](https://pbenavidesh.github.io/psvr/reference/psvr-fit-wrappers.md)
  [`psvr_rmspe_sym_linear_fit()`](https://pbenavidesh.github.io/psvr/reference/psvr-fit-wrappers.md)
  : Fit wrappers for parsnip engine dispatch
