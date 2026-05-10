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

A named list with components `alpha`, `b`, `support_data`. `alpha`
carries `α − α* = β` for `loss = "mape"` and the LS-SVR `α` for
`loss = "rmspe"`.
