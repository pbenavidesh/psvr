# Predict from a fitted epsilon-SVR with MAPE model

Method dispatched on the `"psvr_mape"` class, which both
[`psvr_mape()`](https://pbenavidesh.github.io/psvr/reference/psvr_mape.md)
and the parsnip engine fit wrappers return.

## Usage

``` r
# S3 method for class 'psvr_mape'
predict(object, newdata, ...)
```

## Arguments

- object:

  An object of class `"psvr_mape"`, as returned by
  [`psvr_mape()`](https://pbenavidesh.github.io/psvr/reference/psvr_mape.md)
  with `sym_type = "none"` or by the parsnip engine fit wrappers (see
  [psvr-fit-wrappers](https://pbenavidesh.github.io/psvr/reference/psvr-fit-wrappers.md);
  `sym_type = "even"` or `"odd"` yields `"psvr_mape_sym"` instead).
  Unwrap a parsnip fit with
  [`parsnip::extract_fit_engine()`](https://parsnip.tidymodels.org/reference/reexports.html)
  to obtain it.

- newdata:

  Numeric matrix of new inputs, one observation per row (\\M \times
  p\\).

- ...:

  Ignored.

## Value

Numeric vector of length M with predicted values.
