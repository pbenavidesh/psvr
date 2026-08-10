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

A named list with five components:

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

The LS-SVR classes return **three** components rather than five, since
they have no `alpha_star` and no pruned `beta`; the absent components
are not materialised as `NULL`. So `names(coef(fit))` depends on the
model family. That is deliberate: each class is family-specific, and
inventing empty slots to make the two agree would add structure with
nothing to inherit it from.

## Renamed in 0.0.2.9011

`alpha` previously held the pruned \\\beta\\ and `support_data` was
named `X_sv`, which made `coef(fit)$alpha` mean the length-`n_sv`
\\\beta\\ here but the length-`N` dual \\\alpha\\ on a fit from the
superseded
[`psvr()`](https://pbenavidesh.github.io/psvr/reference/psvr-package.md):
one generic returning two different vectors under one name, silently,
depending on entry point. The \\\beta\\-under-`alpha` meaning is the one
0.0.2.9004 moved away from on the object itself; this aligns
[`coef()`](https://rdrr.io/r/stats/coef.html) with it.
