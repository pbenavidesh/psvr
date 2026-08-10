# Predict from a fitted symmetric epsilon-SVR with MAPE model

Method dispatched on the `"psvr_mape_sym"` class, which both
[`psvr_mape()`](https://pbenavidesh.github.io/psvr/reference/psvr_mape.md)
and the parsnip engine fit wrappers return. Uses the symmetric
representer theorem \$\$f(x) = \tfrac{1}{2}\sum_k \beta_k K_s(x_k, x) +
b\$\$ with \\K_s(x_k, x) = K(x_k, x) + a K(x_k, -x)\\.

## Usage

``` r
# S3 method for class 'psvr_mape_sym'
predict(object, newdata, ...)
```

## Arguments

- object:

  An object of class `"psvr_mape_sym"`, as returned by
  [`psvr_mape()`](https://pbenavidesh.github.io/psvr/reference/psvr_mape.md)
  with `sym_type = "even"` or `"odd"`, or by the parsnip engine fit
  wrappers (see
  [psvr-fit-wrappers](https://pbenavidesh.github.io/psvr/reference/psvr-fit-wrappers.md);
  `sym_type = "none"` yields `"psvr_mape"` instead). Unwrap a parsnip
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
