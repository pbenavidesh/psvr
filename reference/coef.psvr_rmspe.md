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

- `support_data`:

  Training input matrix (all N observations).

The names match
[`coef.psvr_fit()`](https://pbenavidesh.github.io/psvr/reference/coef.psvr_fit.md)
on a `loss = "rmspe"` fit. That method additionally carries `alpha_star`
and `beta` as `NULL`, because one class serves both families; here they
are simply absent, so `$alpha_star` and `$beta` are `NULL` either way.

## Renamed in 0.0.2.9011

`support_data` was named `X_sv`. LS-SVR performs no pruning — every
training point contributes to `f(x)` — so there are no support vectors
to name: it was an epsilon-SVR name on an LS-SVR value. The value is
unchanged.
