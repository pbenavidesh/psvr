# Extract training residuals from a psvr model

Three residual types are available. They are **different quantities with
different denominators**, not interchangeable scalings of one another:

## Usage

``` r
# S3 method for class 'psvr_mape'
residuals(object, type = c("response", "percentage", "multiplicative"), ...)

# S3 method for class 'psvr_mape_sym'
residuals(object, type = c("response", "percentage", "multiplicative"), ...)

# S3 method for class 'psvr_rmspe'
residuals(object, type = c("response", "percentage", "multiplicative"), ...)

# S3 method for class 'psvr_rmspe_sym'
residuals(object, type = c("response", "percentage", "multiplicative"), ...)
```

## Arguments

- object:

  A fitted object of class `"psvr_mape"`, `"psvr_mape_sym"`,
  `"psvr_rmspe"` or `"psvr_rmspe_sym"`, from
  [`psvr_mape()`](https://pbenavidesh.github.io/psvr/reference/psvr_mape.md),
  [`psvr_rmspe()`](https://pbenavidesh.github.io/psvr/reference/psvr_rmspe.md),
  or a parsnip fit unwrapped with
  [`parsnip::extract_fit_engine()`](https://parsnip.tidymodels.org/reference/reexports.html).

- type:

  One of `"response"` (default), `"percentage"`, or `"multiplicative"`.
  See above; the denominators differ.

- ...:

  Ignored.

## Value

Numeric vector of length `N` (the number of training observations), in
training-row order.

## Details

- `"response"`:

  \\y - \hat{y}\\. The default, following the R convention for
  [`stats::residuals()`](https://rdrr.io/r/stats/residuals.html). Units
  of the response.

- `"percentage"`:

  \\(y - \hat{y}) / y\\. Divides by the **observed target**. This is the
  per-observation contribution to the MAPE loss the epsilon-SVR models
  are fitted under, so it is the residual that corresponds to the
  estimated objective. `mean(abs(.)) * 100` is the training MAPE.

- `"multiplicative"`:

  \\(y - \hat{y}) / \hat{y}\\. Divides by the **fitted value**. This is
  the \\\hat{\eta}\\ of the multiplicative-noise model \\Y = f(x)(1 +
  \eta)\\, under which \\\eta = (Y - f(x))/f(x)\\. Use it to inspect the
  assumed noise structure, e.g. checking whether \\\hat{\eta}\\ is
  homoscedastic and centred at zero.

The denominators differ and the choice matters: `"percentage"` divides
by the observed target because that is what the MAPE loss does,
`"multiplicative"` divides by the fitted value because that is what the
noise model does. They agree only when \\y = \hat{y}\\, and diverge as
the fit degrades. Choose by the question being asked: the loss actually
minimised (`"percentage"`), or the noise model assumed
(`"multiplicative"`).

## Near-zero fitted values

`"multiplicative"` divides by \\\hat{y}\\, which an SVR does not
constrain to be positive even though the targets are. Where \\\hat{y}\\
is at or below `sqrt(.Machine$double.eps) * mean(abs(y))` - including
zero and negative fitted values - the ratio is inflated, infinite, or
sign-flipped.

This function does **not** drop those observations: it always returns
exactly `N` values in training-row order, so the result stays aligned
with `y_train`,
[`fitted()`](https://rdrr.io/r/stats/fitted.values.html), and the
training rows. The raw value (possibly `Inf` or `NaN`) is returned and a
single warning reports how many observations are affected.

For *pooled* diagnostics (a mean multiplicative error, a variance, a
histogram) the affected observations must be excluded, or one near-zero
fitted value dominates the summary. This follows the standard treatment
of percentage errors near zero (Makridakis): screen the fitted values
against a threshold and drop the observations below it before pooling,
rather than altering the per-observation values. Exclude at the point of
aggregation, e.g.:

    e <- residuals(fit, type = "multiplicative")
    mean(e[is.finite(e)])

## Not reachable through parsnip

parsnip registers neither `residuals.model_fit` nor `fitted.model_fit`,
so calling either generic on a `model_fit` dispatches to the stats
default and returns `NULL` **silently** - no error, no warning. Reach
the psvr object first with
[`parsnip::extract_fit_engine()`](https://parsnip.tidymodels.org/reference/reexports.html),
then call [`residuals()`](https://rdrr.io/r/stats/residuals.html) or
[`fitted()`](https://rdrr.io/r/stats/fitted.values.html) on that. psvr
deliberately does not register S3 methods on parsnip's class. Note also
that
[`parsnip::augment()`](https://generics.r-lib.org/reference/augment.html)
recomputes predictions on whatever `new_data` it is given and reports
response residuals only, so it is not a substitute for the
`"percentage"` and `"multiplicative"` types.

## See also

[`fitted.psvr_mape()`](https://pbenavidesh.github.io/psvr/reference/psvr-fitted.md)
and the other fitted methods

## Examples

``` r
set.seed(1)
X <- matrix(runif(40, 0.5, 3), 20, 2)
y <- 2 + X[, 1]^2
fit <- psvr_rmspe(X, y, kernel = make_kernel("rbf"), gamma = 100)
head(residuals(fit))
#> [1] -0.30670706 -0.09389727 -0.07896843  0.70298893  0.02611907  0.84129179
mean(abs(residuals(fit, type = "percentage"))) * 100   # training MAPE
#> [1] 4.973158

# Through parsnip both generics return NULL on the model_fit wrapper;
# extract the engine object first.
df   <- data.frame(x1 = X[, 1], x2 = X[, 2], y = y)
spec <- psvr_rmspe_rbf(cost = 10, rbf_sigma = 0.8)
pfit <- parsnip::fit(spec, y ~ x1 + x2, data = df)
residuals(pfit)                                     # NULL
#> NULL
head(residuals(parsnip::extract_fit_engine(pfit)))  # the residuals
#> [1] -0.7450443 -0.1093886  0.3500610  3.1820853 -0.3142471  2.9735249
```
