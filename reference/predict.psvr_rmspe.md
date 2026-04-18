# Predict from a fitted LS-SVR with RMSPE model

Predict from a fitted LS-SVR with RMSPE model

## Usage

``` r
# S3 method for class 'psvr_rmspe'
predict(object, newdata, ...)
```

## Arguments

- object:

  An object of class `"psvr_rmspe"` from
  [`rmspe_lssvr()`](https://pbenavidesh.github.io/psvr/reference/rmspe_lssvr.md).

- newdata:

  Numeric matrix of new inputs, one observation per row (M × p).

- ...:

  Ignored.

## Value

Numeric vector of length M with predicted values.

## Examples

``` r
X <- matrix(c(1, 2, 3, 4, 5, 6), ncol = 2)
y <- c(2.1, 3.8, 6.2)
K <- make_kernel("rbf", sigma = 1)
fit <- rmspe_lssvr(X, y, kernel = K, gamma = 1)
predict(fit, X)
#> [1] 2.743967 2.904833 2.969809
```
