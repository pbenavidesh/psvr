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
psvr_mape_rbf_fit(x, y, C, eps, rbf_sigma = 1, tol = 1e-05)

psvr_mape_poly_fit(x, y, C, eps, degree = 3L, scale_factor = 1, tol = 1e-05)

psvr_mape_linear_fit(x, y, C, eps, tol = 1e-05)

psvr_mape_sym_rbf_fit(
  x,
  y,
  C,
  eps,
  rbf_sigma = 1,
  sym_type = "even",
  tol = 1e-05
)

psvr_mape_sym_poly_fit(
  x,
  y,
  C,
  eps,
  degree = 3L,
  scale_factor = 1,
  a = 1L,
  tol = 1e-05
)

psvr_mape_sym_linear_fit(x, y, C, eps, a = 1L, tol = 1e-05)

psvr_rmspe_rbf_fit(x, y, gamma, rbf_sigma = 1)

psvr_rmspe_poly_fit(x, y, gamma, degree = 3L, scale_factor = 1)

psvr_rmspe_linear_fit(x, y, gamma)

psvr_rmspe_sym_rbf_fit(x, y, gamma, rbf_sigma = 1, sym_type = "even")

psvr_rmspe_sym_poly_fit(x, y, gamma, degree = 3L, scale_factor = 1, a = 1L)

psvr_rmspe_sym_linear_fit(x, y, gamma, a = 1L)
```

## Arguments

- x:

  Numeric predictor matrix (parsnip matrix interface).

- y:

  Numeric outcome vector (strictly positive).

- C:

  Regularization parameter for MAPE models (see
  [`mape_svr()`](https://pbenavidesh.github.io/psvr/reference/mape_svr.md)).

- eps:

  Epsilon tube half-width for MAPE models.

- rbf_sigma:

  RBF bandwidth σ \> 0.

- tol:

  Solver zero-threshold (see
  [`mape_svr()`](https://pbenavidesh.github.io/psvr/reference/mape_svr.md)).

- degree:

  Polynomial degree ≥ 1.

- scale_factor:

  Polynomial constant term (coef₀).

- sym_type:

  Symmetry type (`"even"` or `"odd"`) for symmetric models; translated
  to `a = 1L` or `a = -1L` before calling the solver.

- gamma:

  Regularization parameter for RMSPE models (see
  [`rmspe_lssvr()`](https://pbenavidesh.github.io/psvr/reference/rmspe_lssvr.md)).
