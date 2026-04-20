# Parsnip model specs: epsilon-SVR with MAPE loss (Model 1)

Create parsnip model specifications for
[`mape_svr()`](https://pbenavidesh.github.io/psvr/reference/mape_svr.md)
with a fixed kernel type. Kernel parameters are tunable parsnip
arguments; the symmetry parameter `a` and solver tolerance are engine
arguments passed via `set_engine()`.

## Usage

``` r
psvr_mape_rbf(
  mode = "regression",
  engine = "psvr",
  cost = NULL,
  svm_margin = NULL,
  rbf_sigma = NULL
)

psvr_mape_poly(
  mode = "regression",
  engine = "psvr",
  cost = NULL,
  svm_margin = NULL,
  degree = NULL,
  scale_factor = NULL
)

psvr_mape_linear(
  mode = "regression",
  engine = "psvr",
  cost = NULL,
  svm_margin = NULL
)
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
spec <- psvr_mape_rbf(cost = 10, svm_margin = 1, rbf_sigma = 1) |>
  set_engine("psvr")

spec_poly <- psvr_mape_poly(cost = 10, svm_margin = 1, degree = 2,
                            scale_factor = 1) |>
  set_engine("psvr")

spec_lin <- psvr_mape_linear(cost = 10, svm_margin = 1) |>
  set_engine("psvr")
} # }
```
