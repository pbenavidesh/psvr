# Extract coefficients from a psvr_fit model

Extract coefficients from a psvr_fit model

## Usage

``` r
# S3 method for class 'psvr_fit'
coef(object, ...)
```

## Arguments

- object:

  An object of class `"psvr_fit"`.

- ...:

  Ignored.

## Value

A named list. For `loss = "mape"` it contains `alpha` and `alpha_star`
(the length-`N` pre-pruning dual variables), `beta` (the pruned `α − α*`
of length `n_sv` used by
[`predict()`](https://rdrr.io/r/stats/predict.html)), `b`, and
`support_data`. For `loss = "rmspe"`, `alpha` (length `N`, the LS-SVR
solution), `b`, and `support_data` (with `alpha_star` and `beta` set to
`NULL`).
