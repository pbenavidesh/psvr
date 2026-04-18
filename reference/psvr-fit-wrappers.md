# Fit wrappers for parsnip engine dispatch

Low-level bridge functions called by parsnip when fitting psvr model
specs. Not intended for direct use; call
[`mape_svr()`](https://pbenavidesh.github.io/psvr/reference/mape_svr.md),
[`mape_sym_svr()`](https://pbenavidesh.github.io/psvr/reference/mape_sym_svr.md),
[`rmspe_lssvr()`](https://pbenavidesh.github.io/psvr/reference/rmspe_lssvr.md),
or
[`rmspe_sym_lssvr()`](https://pbenavidesh.github.io/psvr/reference/rmspe_sym_lssvr.md)
directly instead.

## Usage

``` r
psvr_mape_fit(x, y, kernel, C, eps, tol = 1e-05)

psvr_mape_sym_fit(x, y, kernel, C, eps, a = 1L, tol = 1e-05)

psvr_rmspe_fit(x, y, kernel, gamma)

psvr_rmspe_sym_fit(x, y, kernel, gamma, a = 1L)
```

## Arguments

- x:

  Numeric predictor matrix (parsnip matrix interface).

- y:

  Numeric outcome vector.

- kernel:

  A kernel function from
  [`make_kernel()`](https://pbenavidesh.github.io/psvr/reference/make_kernel.md).

- C, eps:

  Hyperparameters for MAPE models (see
  [`mape_svr()`](https://pbenavidesh.github.io/psvr/reference/mape_svr.md)).

- tol:

  Solver zero-threshold (see
  [`mape_svr()`](https://pbenavidesh.github.io/psvr/reference/mape_svr.md)).

- a:

  Symmetry parameter (`1` or `-1`) for symmetric models.

- gamma:

  Regularization parameter for RMSPE models (see
  [`rmspe_lssvr()`](https://pbenavidesh.github.io/psvr/reference/rmspe_lssvr.md)).
