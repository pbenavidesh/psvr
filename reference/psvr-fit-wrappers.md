# Fit wrappers for parsnip engine dispatch

Bridge functions called by parsnip when fitting psvr model specs.
Exported only because parsnip's resolver requires it; not intended for
direct use. Call
[`psvr_mape()`](https://pbenavidesh.github.io/psvr/reference/psvr_mape.md)
or
[`psvr_rmspe()`](https://pbenavidesh.github.io/psvr/reference/psvr_rmspe.md)
instead for direct fitting.

## Usage

``` r
psvr_mape_rbf_fit(
  x,
  y,
  C,
  eps,
  rbf_sigma = 1,
  sym_type = "none",
  tol = 0.001,
  max_iter = 100000L
)

psvr_mape_poly_fit(
  x,
  y,
  C,
  eps,
  degree = 3L,
  scale_factor = 1,
  sym_type = "none",
  tol = 0.001,
  max_iter = 100000L
)

psvr_mape_linear_fit(
  x,
  y,
  C,
  eps,
  sym_type = "none",
  tol = 0.001,
  max_iter = 100000L
)

psvr_rmspe_rbf_fit(
  x,
  y,
  gamma,
  rbf_sigma = 1,
  sym_type = "none",
  precondition = "auto"
)

psvr_rmspe_poly_fit(
  x,
  y,
  gamma,
  degree = 3L,
  scale_factor = 1,
  sym_type = "none",
  precondition = "auto"
)

psvr_rmspe_linear_fit(x, y, gamma, sym_type = "none", precondition = "auto")
```

## Arguments

- x:

  Numeric predictor matrix (parsnip matrix interface).

- y:

  Numeric outcome vector (strictly positive).

- C:

  Regularization parameter for MAPE models.

- eps:

  Epsilon tube half-width for MAPE models.

- rbf_sigma:

  RBF bandwidth \\\sigma \> 0\\.

- sym_type:

  Symmetry type. `"none"` (the default) dispatches to the non-symmetric
  fitter; `"even"` and `"odd"` dispatch to the symmetric fitter with
  `a = 1L` and `a = -1L` respectively.

- tol:

  Solver convergence tolerance for the SMO loop. Default `1e-3`.

- max_iter:

  Maximum SMO iterations. Default `100000L`. The solver emits a
  [`warning()`](https://rdrr.io/r/base/warning.html) and returns
  `converged = FALSE` if it does not converge within `max_iter`.

- degree:

  Polynomial degree \\\ge 1\\.

- scale_factor:

  Polynomial constant term (\\\mathrm{coef}\_0\\).

- gamma:

  Regularization parameter for RMSPE models.

- precondition:

  Optional symmetric rescaling preconditioner for the RMSPE LS-SVR
  fitters. See
  [`psvr_rmspe()`](https://pbenavidesh.github.io/psvr/reference/psvr_rmspe.md)
  for accepted values and semantics.

## Value

A fitted model object of the S3 class matching the wrapper's model
family, returned unmodified from the internal fitter. These are the same
classes
[`psvr_mape()`](https://pbenavidesh.github.io/psvr/reference/psvr_mape.md)
and
[`psvr_rmspe()`](https://pbenavidesh.github.io/psvr/reference/psvr_rmspe.md)
return, so a parsnip fit unwrapped with
[`parsnip::extract_fit_engine()`](https://parsnip.tidymodels.org/reference/reexports.html)
and a directly fitted object are interchangeable. **Which class is
returned depends on `sym_type`**, since each wrapper dispatches to the
symmetric or non-symmetric fitter.

The MAPE wrappers (`psvr_mape_rbf_fit()`, `psvr_mape_poly_fit()`,
`psvr_mape_linear_fit()`) with `sym_type = "none"` return an object of
class `"psvr_mape"`: a list with `beta` (support-vector dual
differences), `alpha` and `alpha_star` (length-`N` pre-pruning duals,
retained for warm starts), `b`, `X_sv`, `y_sv`, `y_train`,
`fitted_values`, `kernel`, `C`, `eps`, `n_train`, `p_train`,
`iterations`, `converged`, and `block_k4`. With `sym_type = "even"` or
`"odd"` they return class `"psvr_mape_sym"`: the same components plus
`a` (the symmetry type) and `spectral` (Algorithm 2 diagnostics).

The RMSPE wrappers (`psvr_rmspe_rbf_fit()`, `psvr_rmspe_poly_fit()`,
`psvr_rmspe_linear_fit()`) with `sym_type = "none"` return class
`"psvr_rmspe"`: a list with `alpha`, `b`, `X_train`, `y_train`,
`fitted_values`, `kernel`, `gamma`, `n_train`, `p_train`, and
`precondition_applied`. With `sym_type = "even"` or `"odd"` they return
class `"psvr_rmspe_sym"`: the same components plus `a`.
