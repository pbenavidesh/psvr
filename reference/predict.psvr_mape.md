# Predict from a fitted epsilon-SVR with MAPE model

Method dispatched on the legacy `"psvr_mape"` class returned by the
deprecated
[`mape_svr()`](https://pbenavidesh.github.io/psvr/reference/mape_svr.md).
New code should use
[`psvr()`](https://pbenavidesh.github.io/psvr/reference/psvr.md) (which
returns a `"psvr_fit"` object dispatched by
[`predict.psvr_fit()`](https://pbenavidesh.github.io/psvr/reference/predict.psvr_fit.md)).

## Usage

``` r
# S3 method for class 'psvr_mape'
predict(object, newdata, ...)
```

## Arguments

- object:

  An object of class `"psvr_mape"` from
  [`mape_svr()`](https://pbenavidesh.github.io/psvr/reference/mape_svr.md).

- newdata:

  Numeric matrix of new inputs, one observation per row (M × p).

- ...:

  Ignored.

## Value

Numeric vector of length M with predicted values.
