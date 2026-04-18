# Extract coefficients from a psvr_mape model

Extract coefficients from a psvr_mape model

## Usage

``` r
# S3 method for class 'psvr_mape'
coef(object, ...)
```

## Arguments

- object:

  An object of class `"psvr_mape"`.

- ...:

  Ignored.

## Value

A named list with components:

- `alpha`:

  Dual variable differences `βk = αk − αk*` for support vectors.

- `b`:

  Bias term.

- `X_sv`:

  Support vector input matrix.
