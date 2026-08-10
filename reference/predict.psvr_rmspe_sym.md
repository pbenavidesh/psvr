# Predict from a fitted symmetric LS-SVR with RMSPE model

Method dispatched on the `"psvr_rmspe_sym"` class, which both
[`psvr_rmspe()`](https://pbenavidesh.github.io/psvr/reference/psvr_rmspe.md)
and the parsnip engine fit wrappers return. Uses the symmetric
representer \$\$f(x) = \sum_k \alpha_k \cdot \tfrac{1}{2}\left(K(x_k,
x) + a K(x_k, -x)\right) + b\$\$

## Usage

``` r
# S3 method for class 'psvr_rmspe_sym'
predict(object, newdata, ...)
```

## Arguments

- object:

  An object of class `"psvr_rmspe_sym"`, as returned by
  [`psvr_rmspe()`](https://pbenavidesh.github.io/psvr/reference/psvr_rmspe.md)
  with `sym_type = "even"` or `"odd"`, or by the parsnip engine fit
  wrappers (see
  [psvr-fit-wrappers](https://pbenavidesh.github.io/psvr/reference/psvr-fit-wrappers.md);
  `sym_type = "none"` yields `"psvr_rmspe"` instead). Unwrap a parsnip
  fit with
  [`parsnip::extract_fit_engine()`](https://parsnip.tidymodels.org/reference/reexports.html)
  to obtain it.

- newdata:

  Numeric matrix of new inputs, one observation per row (\\M \times
  p\\).

- ...:

  Ignored.

## Value

Numeric vector of length M with predicted values.
