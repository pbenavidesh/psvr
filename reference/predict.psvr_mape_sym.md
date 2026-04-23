# Predict from a fitted symmetric epsilon-SVR with MAPE model

Uses the symmetric representer theorem: `f(x) = ½ Σk βk Ks(xk, x) + b`
where `Ks(xk, x) = K(xk, x) + a·K(xk, -x)`.

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

## Examples

``` r
X <- matrix(1:6, ncol = 2)
y <- c(2.1, 3.8, 6.2)
K <- make_kernel("rbf", sigma = 1)
fit <- mape_sym_svr(X, y, kernel = K, C = 10, eps = 5, a = 1)
predict(fit, X)
#> [1] 2.205002 3.990000 5.889998
```
