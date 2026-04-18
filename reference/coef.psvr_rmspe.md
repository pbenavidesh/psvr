# Extract coefficients from a psvr_rmspe model

Extract coefficients from a psvr_rmspe model

## Usage

``` r
# S3 method for class 'psvr_rmspe'
coef(object, ...)
```

## Arguments

- object:

  An object of class `"psvr_rmspe"`.

- ...:

  Ignored.

## Value

A named list with components:

- `alpha`:

  Dual variables / Lagrange multipliers (length N).

- `b`:

  Bias term.

- `X_sv`:

  Training input matrix (all N observations).
