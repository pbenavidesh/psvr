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

- `support_data`:

  Training input matrix (all N observations).

Three components, not the five the MAPE classes return: LS-SVR has no
`alpha_star` and no pruned `beta`, and they are not materialised as
`NULL`. See
[`coef.psvr_rmspe()`](https://pbenavidesh.github.io/psvr/reference/coef.psvr_rmspe.md).

## Renamed in 0.0.2.9011

See
[`coef.psvr_rmspe()`](https://pbenavidesh.github.io/psvr/reference/coef.psvr_rmspe.md)
— the same rename, for the same reason, applied to both LS-SVR classes
together.
