# Predict from a fitted epsilon-SVR with MAPE model

Method dispatched on the legacy `"psvr_mape"` class, which the parsnip
engine fit wrappers return. For direct fitting use
[`psvr()`](https://pbenavidesh.github.io/psvr/reference/psvr.md), which
returns a `"psvr_fit"` object dispatched by
[`predict.psvr_fit()`](https://pbenavidesh.github.io/psvr/reference/predict.psvr_fit.md).

## Usage

``` r
# S3 method for class 'psvr_mape'
predict(object, newdata, ...)
```

## Arguments

- object:

  An object of class `"psvr_mape"`, as returned by the parsnip engine
  fit wrappers with `sym_type = "none"` (see
  [psvr-fit-wrappers](https://pbenavidesh.github.io/psvr/reference/psvr-fit-wrappers.md);
  `sym_type = "even"` or `"odd"` yields `"psvr_mape_sym"` instead).
  Unwrap a parsnip fit with
  [`parsnip::extract_fit_engine()`](https://parsnip.tidymodels.org/reference/reexports.html)
  to obtain it.

- newdata:

  Numeric matrix of new inputs, one observation per row (M × p).

- ...:

  Ignored.

## Value

Numeric vector of length M with predicted values.
