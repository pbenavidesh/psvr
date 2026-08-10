# Extract coefficients from a psvr_mape_sym model

Extract coefficients from a psvr_mape_sym model

## Usage

``` r
# S3 method for class 'psvr_mape_sym'
coef(object, ...)
```

## Arguments

- object:

  An object of class `"psvr_mape_sym"`.

- ...:

  Ignored.

## Value

A named list with five components, identical in meaning to
[`coef.psvr_mape()`](https://pbenavidesh.github.io/psvr/reference/coef.psvr_mape.md):

- `alpha`, `alpha_star`:

  The length-`N` pre-pruning dual variables \\\alpha_k\\ and
  \\\alpha^\*\_k\\.

- `beta`:

  The pruned dual differences \\\beta_k = \alpha_k - \alpha^\*\_k\\ over
  the support-vector indices (length `n_sv`); this is what
  [`predict()`](https://rdrr.io/r/stats/predict.html) uses.

- `b`:

  Bias term.

- `support_data`:

  Support vector input matrix.

## Renamed in 0.0.2.9011

See
[`coef.psvr_mape()`](https://pbenavidesh.github.io/psvr/reference/coef.psvr_mape.md)
— the same rename, for the same reason, applied to both MAPE classes
together.
