# Package index

## Model fitting

One fitter per model family.
[`psvr_mape()`](https://pbenavidesh.github.io/psvr/reference/psvr_mape.md)
covers the epsilon-SVR models (1 and 2),
[`psvr_rmspe()`](https://pbenavidesh.github.io/psvr/reference/psvr_rmspe.md)
the LS-SVR models (3 and 4); within each,
`sym_type = "none" | "even" | "odd"` selects the symmetric variant. They
are separate functions because the two families share no solver, no dual
structure and no hyperparameter search space.

- [`psvr_mape()`](https://pbenavidesh.github.io/psvr/reference/psvr_mape.md)
  : Fit an epsilon-SVR with MAPE loss
- [`psvr_rmspe()`](https://pbenavidesh.github.io/psvr/reference/psvr_rmspe.md)
  : Fit a least-squares SVR with RMSPE loss

## Kernel factory

Shared kernel interface used by all four models.

- [`make_kernel()`](https://pbenavidesh.github.io/psvr/reference/make_kernel.md)
  : Create a kernel function

## Cross-validation

Fold-wise fitting for `loss = "mape"`, carrying the converged SMO dual
variables from one fold into the next as a warm start.

- [`psvr_cv()`](https://pbenavidesh.github.io/psvr/reference/psvr_cv.md)
  : Cross-validate psvr_mape() with automatic warm-start across folds

## tidymodels / parsnip interface — ε-SVR with MAPE (Models 1 & 2)

Parsnip model specifications for the ε-SVR family (one per kernel type).
`sym_type = "none"` fits Model 1; `"even"` / `"odd"` fit Model 2.

- [`psvr_mape_rbf()`](https://pbenavidesh.github.io/psvr/reference/psvr_mape_specs.md)
  [`psvr_mape_poly()`](https://pbenavidesh.github.io/psvr/reference/psvr_mape_specs.md)
  [`psvr_mape_linear()`](https://pbenavidesh.github.io/psvr/reference/psvr_mape_specs.md)
  : Parsnip model specs: epsilon-SVR with MAPE loss (Model 1)

## tidymodels / parsnip interface — LS-SVR with RMSPE (Models 3 & 4)

Parsnip model specifications for the LS-SVR family (one per kernel
type). `sym_type = "none"` fits Model 3; `"even"` / `"odd"` fit Model 4.

- [`psvr_rmspe_rbf()`](https://pbenavidesh.github.io/psvr/reference/psvr_rmspe_specs.md)
  [`psvr_rmspe_poly()`](https://pbenavidesh.github.io/psvr/reference/psvr_rmspe_specs.md)
  [`psvr_rmspe_linear()`](https://pbenavidesh.github.io/psvr/reference/psvr_rmspe_specs.md)
  : Parsnip model specs: LS-SVR with RMSPE loss (Model 3)

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

## Fit-object methods

Methods for the four fit classes — `psvr_mape`, `psvr_mape_sym`,
`psvr_rmspe`, `psvr_rmspe_sym`. These are what
[`psvr_mape()`](https://pbenavidesh.github.io/psvr/reference/psvr_mape.md)
and
[`psvr_rmspe()`](https://pbenavidesh.github.io/psvr/reference/psvr_rmspe.md)
return **and** what the parsnip engine fit wrappers return, so a direct
fit and a parsnip fit unwrapped with
[`parsnip::extract_fit_engine()`](https://hardhat.tidymodels.org/reference/hardhat-extract.html)
are the same object.

- [`predict(`*`<psvr_mape>`*`)`](https://pbenavidesh.github.io/psvr/reference/predict.psvr_mape.md)
  : Predict from a fitted epsilon-SVR with MAPE model
- [`predict(`*`<psvr_mape_sym>`*`)`](https://pbenavidesh.github.io/psvr/reference/predict.psvr_mape_sym.md)
  : Predict from a fitted symmetric epsilon-SVR with MAPE model
- [`predict(`*`<psvr_rmspe>`*`)`](https://pbenavidesh.github.io/psvr/reference/predict.psvr_rmspe.md)
  : Predict from a fitted LS-SVR with RMSPE model
- [`predict(`*`<psvr_rmspe_sym>`*`)`](https://pbenavidesh.github.io/psvr/reference/predict.psvr_rmspe_sym.md)
  : Predict from a fitted symmetric LS-SVR with RMSPE model
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
- [`summary(`*`<psvr_mape>`*`)`](https://pbenavidesh.github.io/psvr/reference/summary.psvr_mape.md)
  : Summarize a fitted epsilon-SVR with MAPE loss
- [`summary(`*`<psvr_mape_sym>`*`)`](https://pbenavidesh.github.io/psvr/reference/summary.psvr_mape_sym.md)
  : Summarize a fitted symmetric epsilon-SVR with MAPE loss
- [`summary(`*`<psvr_rmspe>`*`)`](https://pbenavidesh.github.io/psvr/reference/summary.psvr_rmspe.md)
  : Summarize a fitted LS-SVR with RMSPE loss
- [`summary(`*`<psvr_rmspe_sym>`*`)`](https://pbenavidesh.github.io/psvr/reference/summary.psvr_rmspe_sym.md)
  : Summarize a fitted symmetric LS-SVR with RMSPE loss
- [`fitted(`*`<psvr_mape>`*`)`](https://pbenavidesh.github.io/psvr/reference/psvr-fitted.md)
  [`fitted(`*`<psvr_mape_sym>`*`)`](https://pbenavidesh.github.io/psvr/reference/psvr-fitted.md)
  [`fitted(`*`<psvr_rmspe>`*`)`](https://pbenavidesh.github.io/psvr/reference/psvr-fitted.md)
  [`fitted(`*`<psvr_rmspe_sym>`*`)`](https://pbenavidesh.github.io/psvr/reference/psvr-fitted.md)
  : Extract training fitted values from a psvr model
- [`residuals(`*`<psvr_mape>`*`)`](https://pbenavidesh.github.io/psvr/reference/psvr-residuals.md)
  [`residuals(`*`<psvr_mape_sym>`*`)`](https://pbenavidesh.github.io/psvr/reference/psvr-residuals.md)
  [`residuals(`*`<psvr_rmspe>`*`)`](https://pbenavidesh.github.io/psvr/reference/psvr-residuals.md)
  [`residuals(`*`<psvr_rmspe_sym>`*`)`](https://pbenavidesh.github.io/psvr/reference/psvr-residuals.md)
  : Extract training residuals from a psvr model
