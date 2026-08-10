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

Three components, against five for the MAPE classes
([`coef.psvr_mape()`](https://pbenavidesh.github.io/psvr/reference/coef.psvr_mape.md)):
LS-SVR has no `alpha_star` and no pruned `beta`, and they are **not**
materialised as `NULL`. So `names(coef(fit))` depends on the model
family, which is a decision rather than an oversight – each class is
family-specific, and inventing empty slots to make the two agree would
add structure with nothing to inherit it from. `$alpha_star` and `$beta`
yield `NULL` on both, so every accessor still agrees.

## Renamed in 0.0.2.9011

`support_data` was named `X_sv`. LS-SVR performs no pruning — every
training point contributes to `f(x)` — so there are no support vectors
to name: it was an epsilon-SVR name on an LS-SVR value. The value is
unchanged.
