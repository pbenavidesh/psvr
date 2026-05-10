# Predict from a fitted LS-SVR with RMSPE model

Method dispatched on the legacy `"psvr_rmspe"` class returned by the
deprecated
[`rmspe_lssvr()`](https://pbenavidesh.github.io/psvr/reference/rmspe_lssvr.md).
New code should use
[`psvr()`](https://pbenavidesh.github.io/psvr/reference/psvr.md).

## Usage

``` r
# S3 method for class 'psvr_rmspe'
predict(object, newdata, ...)
```

## Arguments

- object:

  An object of class `"psvr_rmspe"` from
  [`rmspe_lssvr()`](https://pbenavidesh.github.io/psvr/reference/rmspe_lssvr.md).

- newdata:

  Numeric matrix of new inputs, one observation per row (M × p).

- ...:

  Ignored.

## Value

Numeric vector of length M with predicted values.
