# Parsnip model specs: LS-SVR with RMSPE loss (Model 3)

Create parsnip model specifications for
[`rmspe_lssvr()`](https://pbenavidesh.github.io/psvr/reference/rmspe_lssvr.md)
with a fixed kernel type. `cost` maps to the regularization parameter
`Γ`.

## Usage

``` r
psvr_rmspe_rbf(
  mode = "regression",
  engine = "psvr",
  cost = NULL,
  rbf_sigma = NULL
)

psvr_rmspe_poly(
  mode = "regression",
  engine = "psvr",
  cost = NULL,
  degree = NULL,
  scale_factor = NULL
)

psvr_rmspe_linear(mode = "regression", engine = "psvr", cost = NULL)
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

- rbf_sigma:

  RBF bandwidth σ \> 0. Use
  [`hardhat::tune()`](https://hardhat.tidymodels.org/reference/tune.html)
  to optimize. (RBF specs only.)

- degree:

  Polynomial degree ≥ 1. Use
  [`hardhat::tune()`](https://hardhat.tidymodels.org/reference/tune.html)
  to optimize. (Polynomial specs only.)

- scale_factor:

  Polynomial constant term (coef₀). Use
  [`hardhat::tune()`](https://hardhat.tidymodels.org/reference/tune.html)
  to optimize. (Polynomial specs only.)

## Value

A parsnip `model_spec` object of the corresponding class.

## Examples

``` r
if (FALSE) { # \dontrun{
library(parsnip)
spec <- psvr_rmspe_rbf(cost = 1000, rbf_sigma = 1) |>
  set_engine("psvr")

spec_poly <- psvr_rmspe_poly(cost = 1000, degree = 2, scale_factor = 1) |>
  set_engine("psvr")

spec_lin <- psvr_rmspe_linear(cost = 1000) |>
  set_engine("psvr")
} # }
```
