# Predict from a fitted symmetric LS-SVR with RMSPE model

Method dispatched on the legacy `"psvr_rmspe_sym"` class returned by the
deprecated
[`rmspe_sym_lssvr()`](https://pbenavidesh.github.io/psvr/reference/rmspe_sym_lssvr.md).
Uses the symmetric representer
`f(x) = Σₖ αₖ · ½(K(xₖ, x) + a·K(xₖ, -x)) + b`. New code should use
[`psvr()`](https://pbenavidesh.github.io/psvr/reference/psvr.md).

## Usage

``` r
# S3 method for class 'psvr_rmspe_sym'
predict(object, newdata, ...)
```

## Arguments

- object:

  An object of class `"psvr_rmspe_sym"` from
  [`rmspe_sym_lssvr()`](https://pbenavidesh.github.io/psvr/reference/rmspe_sym_lssvr.md).

- newdata:

  Numeric matrix of new inputs, one observation per row (M × p).

- ...:

  Ignored.

## Value

Numeric vector of length M with predicted values.
