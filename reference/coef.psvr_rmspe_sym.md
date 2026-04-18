# Extract coefficients from a psvr_rmspe_sym model

Extract coefficients from a psvr_rmspe_sym model

## Usage

``` r
# S3 method for class 'psvr_rmspe_sym'
coef(object, ...)
```

## Arguments

- object:

  An object of class `"psvr_rmspe_sym"`.

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
