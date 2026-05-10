# Predict from a fitted psvr_fit model

Predict from a fitted psvr_fit model

## Usage

``` r
# S3 method for class 'psvr_fit'
predict(object, newdata, ...)
```

## Arguments

- object:

  An object of class `"psvr_fit"` from
  [`psvr()`](https://pbenavidesh.github.io/psvr/reference/psvr.md).

- newdata:

  Numeric matrix of new inputs, one observation per row (M × p).

- ...:

  Ignored.

## Value

Numeric vector of length M with predicted values.

## Examples

``` r
set.seed(1)
X <- matrix(rnorm(40), 20, 2)
y <- rlnorm(20)
fit <- psvr(X, y, loss = "rmspe", kernel = make_kernel("rbf", sigma = 1),
            gamma = 100)
predict(fit, X[1:3, , drop = FALSE])
#> [1] 0.9189556 0.7369731 0.8524386
```
