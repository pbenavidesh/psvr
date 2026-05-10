# Predict from a fitted symmetric epsilon-SVR with MAPE model

Method dispatched on the legacy `"psvr_mape_sym"` class returned by the
deprecated
[`mape_sym_svr()`](https://pbenavidesh.github.io/psvr/reference/mape_sym_svr.md).
Uses the symmetric representer theorem `f(x) = ½ Σk βk Ks(xk, x) + b`
with `Ks(xk, x) = K(xk, x) + a·K(xk, -x)`. New code should use
[`psvr()`](https://pbenavidesh.github.io/psvr/reference/psvr.md).

## Usage

``` r
# S3 method for class 'psvr_mape_sym'
predict(object, newdata, ...)
```

## Arguments

- object:

  An object of class `"psvr_mape_sym"` from
  [`mape_sym_svr()`](https://pbenavidesh.github.io/psvr/reference/mape_sym_svr.md).

- newdata:

  Numeric matrix of new inputs, one observation per row (M × p).

- ...:

  Ignored.

## Value

Numeric vector of length M with predicted values.
