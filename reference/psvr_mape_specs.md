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
  to optimize. Mapped to
  [`cost_psvr()`](https://pbenavidesh.github.io/psvr/reference/cost_psvr.md)
  with range `[-2, 10]` on the log2 scale — wider than
  [`dials::cost()`](https://dials.tidymodels.org/reference/cost.html) to
  cover the larger values needed by LS-SVR models.

- svm_margin:

  Epsilon tube half-width `ε ≥ 0` expressed as a percentage of each
  target value. Use
  [`hardhat::tune()`](https://hardhat.tidymodels.org/reference/tune.html)
  to optimize. Mapped to
  [`margin_percentage()`](https://pbenavidesh.github.io/psvr/reference/margin_percentage.md)
  with default range `[1, 20]` (percentage units).

- rbf_sigma:

  RBF bandwidth σ \> 0. Use
  [`hardhat::tune()`](https://hardhat.tidymodels.org/reference/tune.html)
  to optimize. Mapped to
  [`rbf_sigma_psvr()`](https://pbenavidesh.github.io/psvr/reference/rbf_sigma_psvr.md);
  the search range auto-finalizes using the median-distance heuristic
  when training data are available. (RBF specs only.)

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
