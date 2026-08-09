# Extract training fitted values from a psvr model

Returns the length-`N` in-sample predictions \\f(x_k)\\ recorded when
the model was fitted. No kernel matrix is rebuilt: the values are
recovered from state the solver already holds. For the MAPE models that
is a matvec against the retained \\\Omega\\; for the LS-SVR models it is
the KKT stationarity identity \$\$f(x_k) = y_k - (10^{-6} +
y_k^2/\Gamma)\\\alpha_k\$\$ which costs \\O(N)\\ and holds in both
preconditioner branches. The training inputs `X` are not retained for
this purpose.

## Usage

``` r
# S3 method for class 'psvr_fit'
fitted(object, ...)

# S3 method for class 'psvr_mape'
fitted(object, ...)

# S3 method for class 'psvr_mape_sym'
fitted(object, ...)

# S3 method for class 'psvr_rmspe'
fitted(object, ...)

# S3 method for class 'psvr_rmspe_sym'
fitted(object, ...)
```

## Arguments

- object:

  A `"psvr_fit"` object from
  [`psvr()`](https://pbenavidesh.github.io/psvr/reference/psvr.md), or
  one of the legacy `"psvr_mape"`, `"psvr_mape_sym"`, `"psvr_rmspe"`,
  `"psvr_rmspe_sym"` objects from the deprecated fitters.

- ...:

  Ignored.

## Value

Numeric vector of length `N` (the number of training observations), in
training-row order.

## Details

The result equals `predict(object, X_train)` to machine precision. It is
not bit-identical: the two use different summation orders (a BLAS matvec
versus the column-wise reduction in
[`predict()`](https://rdrr.io/r/stats/predict.html)), and for the LS-SVR
models the identity above is exact only up to the residual of the linear
solve. Observed agreement is within `3e-12` relative across the four
models.

## Not reachable through parsnip

parsnip registers neither `residuals.model_fit` nor `fitted.model_fit`,
so calling either generic on a `model_fit` dispatches to the stats
default and returns `NULL` **silently** - no error, no warning. Reach
the psvr object first with
[`parsnip::extract_fit_engine()`](https://parsnip.tidymodels.org/reference/reexports.html),
then call [`fitted()`](https://rdrr.io/r/stats/fitted.values.html) or
[`residuals()`](https://rdrr.io/r/stats/residuals.html) on that. psvr
deliberately does not register S3 methods on parsnip's class. Note also
that
[`parsnip::augment()`](https://generics.r-lib.org/reference/augment.html)
recomputes predictions on whatever `new_data` it is given and reports
response residuals only.

## See also

[`residuals.psvr_fit()`](https://pbenavidesh.github.io/psvr/reference/residuals.psvr_fit.md)

## Examples

``` r
set.seed(1)
X <- matrix(runif(40, 0.5, 3), 20, 2)
y <- 2 + X[, 1]^2
fit <- psvr(X, y, loss = "rmspe", kernel = make_kernel("rbf"), gamma = 100)
head(fitted(fit))
#> [1] 3.661072 4.139683 5.812108 8.972789 2.982308 8.699083

# Through parsnip both generics return NULL on the model_fit wrapper;
# extract the engine object first.
df   <- data.frame(x1 = X[, 1], x2 = X[, 2], y = y)
spec <- psvr_rmspe_rbf(cost = 10, rbf_sigma = 0.8)
pfit <- parsnip::fit(spec, y ~ x1 + x2, data = df)
fitted(pfit)                                     # NULL
#> NULL
head(fitted(parsnip::extract_fit_engine(pfit)))  # the fitted values
#> [1] 4.099409 4.155175 5.383078 6.493693 3.322674 6.566849
```
