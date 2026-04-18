# Parsnip model spec: LS-SVR with RMSPE loss (Model 3)

Creates a parsnip model specification for
[`rmspe_lssvr()`](https://pbenavidesh.github.io/psvr/reference/rmspe_lssvr.md).
The kernel is an engine argument passed via
[`set_engine()`](https://parsnip.tidymodels.org/reference/set_engine.html).

## Usage

``` r
psvr_rmspe(mode = "regression", engine = "psvr", cost = NULL)
```

## Arguments

- mode:

  Only `"regression"` is supported.

- engine:

  Only `"psvr"` is available.

- cost:

  Regularization parameter `Γ > 0`. Use
  [`hardhat::tune()`](https://hardhat.tidymodels.org/reference/tune.html)
  to optimize.

## Value

A `psvr_rmspe_model` / `model_spec` object.

## Examples

``` r
if (FALSE) { # \dontrun{
library(parsnip)
K <- make_kernel("rbf", sigma = 1)
spec <- psvr_rmspe(cost = 1) |>
  set_engine("psvr", kernel = K)
fit(spec, mpg ~ ., data = mtcars)
} # }
```
