# Predict from a fitted symmetric epsilon-SVR with MAPE model

Method dispatched on the legacy `"psvr_mape_sym"` class, which the
parsnip engine fit wrappers return. Uses the symmetric representer
theorem `f(x) = ½ Σk βk Ks(xk, x) + b` with
`Ks(xk, x) = K(xk, x) + a·K(xk, -x)`. For direct fitting use
[`psvr()`](https://pbenavidesh.github.io/psvr/reference/psvr.md).

## Usage

``` r
# S3 method for class 'psvr_mape_sym'
predict(object, newdata, ...)
```

## Arguments

- object:

  An object of class `"psvr_mape_sym"`, as returned by the parsnip
  engine fit wrappers with `sym_type = "even"` or `"odd"` (see
  [psvr-fit-wrappers](https://pbenavidesh.github.io/psvr/reference/psvr-fit-wrappers.md);
  `sym_type = "none"` yields `"psvr_mape"` instead). Unwrap a parsnip
  fit with
  [`parsnip::extract_fit_engine()`](https://parsnip.tidymodels.org/reference/reexports.html)
  to obtain it.

- newdata:

  Numeric matrix of new inputs, one observation per row (M × p).

- ...:

  Ignored.

## Value

Numeric vector of length M with predicted values.
