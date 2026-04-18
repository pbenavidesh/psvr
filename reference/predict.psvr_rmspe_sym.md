# Predict from a fitted symmetric LS-SVR with RMSPE model

Prediction uses the symmetric representer:
`f(x) = Σₖ αₖ · ½(K(xₖ, x) + a·K(xₖ, -x)) + b`.

## Usage

``` r
# S3 method for class 'psvr_rmspe_sym'
predict(object, newdata, ...)
```

## Arguments

- object:

  An object of class `"psvr_rmspe_sym"` from
  [`rmspe_sym_lssvr()`](https://pbenavidesh.github.io/psvr/reference/rmspe_sym_lssvr.md).

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
fit <- rmspe_sym_lssvr(X, y, kernel = K, gamma = 1, a = 1)
predict(fit, X)
#> [1] 2.769355 2.853120 2.886170
```
