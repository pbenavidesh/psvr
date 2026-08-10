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

A named list with the same component names and meanings as
[`coef.psvr_fit()`](https://pbenavidesh.github.io/psvr/reference/coef.psvr_fit.md)
on a `loss = "mape"` fit, so the two entry points agree:

- `alpha`, `alpha_star`:

  The length-`N` pre-pruning dual variables `αk` and `αk*`.

- `beta`:

  The pruned dual differences `βk = αk − αk*` over the support-vector
  indices (length `n_sv`); this is what
  [`predict()`](https://rdrr.io/r/stats/predict.html) uses.

- `b`:

  Bias term.

- `support_data`:

  Support vector input matrix.

## Renamed in 0.0.2.9011

`alpha` previously held the pruned `β` and `support_data` was named
`X_sv`, which made `coef(fit)$alpha` mean the length-`n_sv` `β` here but
the length-`N` dual `α` on a
[`psvr()`](https://pbenavidesh.github.io/psvr/reference/psvr.md) fit —
the same generic returning two different vectors under one name,
silently, depending on entry point. The `β`-under-`alpha` meaning is the
one 0.0.2.9004 moved away from on the object itself; this aligns
[`coef()`](https://rdrr.io/r/stats/coef.html) with it.
