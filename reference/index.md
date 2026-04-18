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

## tidymodels / parsnip interface

Parsnip model specifications for use in tidymodels workflows.

- [`psvr_mape()`](https://pbenavidesh.github.io/psvr/reference/psvr_mape.md)
  : Parsnip model spec: epsilon-SVR with MAPE loss (Model 1)
- [`psvr_mape_sym()`](https://pbenavidesh.github.io/psvr/reference/psvr_mape_sym.md)
  : Parsnip model spec: symmetric epsilon-SVR with MAPE loss (Model 2)
- [`psvr_rmspe()`](https://pbenavidesh.github.io/psvr/reference/psvr_rmspe.md)
  : Parsnip model spec: LS-SVR with RMSPE loss (Model 3)
- [`psvr_rmspe_sym()`](https://pbenavidesh.github.io/psvr/reference/psvr_rmspe_sym.md)
  : Parsnip model spec: symmetric LS-SVR with RMSPE loss (Model 4)
- [`psvr_mape_fit()`](https://pbenavidesh.github.io/psvr/reference/psvr-fit-wrappers.md)
  [`psvr_mape_sym_fit()`](https://pbenavidesh.github.io/psvr/reference/psvr-fit-wrappers.md)
  [`psvr_rmspe_fit()`](https://pbenavidesh.github.io/psvr/reference/psvr-fit-wrappers.md)
  [`psvr_rmspe_sym_fit()`](https://pbenavidesh.github.io/psvr/reference/psvr-fit-wrappers.md)
  : Fit wrappers for parsnip engine dispatch
