# Predict from a fitted epsilon-SVR with MAPE model

Predict from a fitted epsilon-SVR with MAPE model

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

## Examples

``` r
X <- matrix(1:6, ncol = 2)
y <- c(2.1, 3.8, 6.2)
K <- make_kernel("rbf", sigma = 1)
fit <- mape_svr(X, y, kernel = K, C = 10, eps = 5)
predict(fit, X)
#> [1] 2.207585 3.988709 5.888707
```
