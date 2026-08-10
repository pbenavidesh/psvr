# Predict from a fitted LS-SVR with RMSPE model

Method dispatched on the `"psvr_rmspe"` class, which both
[`psvr_rmspe()`](https://pbenavidesh.github.io/psvr/reference/psvr_rmspe.md)
and the parsnip engine fit wrappers return.

## Usage

``` r
# S3 method for class 'psvr_rmspe'
predict(object, newdata, ...)
```

## Arguments

- object:

  An object of class `"psvr_rmspe"`, as returned by
  [`psvr_rmspe()`](https://pbenavidesh.github.io/psvr/reference/psvr_rmspe.md)
  with `sym_type = "none"` or by the parsnip engine fit wrappers (see
  [psvr-fit-wrappers](https://pbenavidesh.github.io/psvr/reference/psvr-fit-wrappers.md);
  `sym_type = "even"` or `"odd"` yields `"psvr_rmspe_sym"` instead).
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
