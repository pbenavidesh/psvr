# Parsnip model spec: epsilon-SVR with MAPE loss (Model 1)

Creates a parsnip model specification for
[`mape_svr()`](https://pbenavidesh.github.io/psvr/reference/mape_svr.md).
The kernel and solver tolerance are engine arguments passed via
`set_engine()`.

## Usage

``` r
psvr_mape(mode = "regression", engine = "psvr", cost = NULL, svm_margin = NULL)
```

## Arguments

- mode:

  Only `"regression"` is supported.

- engine:

  Only `"psvr"` is available.

- cost:

  Regularization parameter `C > 0`. Use
  [`hardhat::tune()`](https://hardhat.tidymodels.org/reference/tune.html)
  to optimize.

- svm_margin:

  Epsilon tube half-width `ε ≥ 0` (percentage units). Use
  [`hardhat::tune()`](https://hardhat.tidymodels.org/reference/tune.html)
  to optimize.

## Value

A `psvr_mape_model` / `model_spec` object.

## Examples

``` r
if (FALSE) { # \dontrun{
library(parsnip)
K <- make_kernel("rbf", sigma = 1)
spec <- psvr_mape(cost = 10, svm_margin = 5) |>
  set_engine("psvr", kernel = K)
fit(spec, mpg ~ ., data = mtcars)
} # }
```
