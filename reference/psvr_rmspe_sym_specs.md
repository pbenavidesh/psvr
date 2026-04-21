# Parsnip model specs: symmetric LS-SVR with RMSPE loss (Model 4)

Create parsnip model specifications for
[`rmspe_sym_lssvr()`](https://pbenavidesh.github.io/psvr/reference/rmspe_sym_lssvr.md)
with a fixed kernel type. Even symmetry (`a = 1L`) is the default and
does not need to be specified in
[`set_engine()`](https://parsnip.tidymodels.org/reference/set_engine.html).
Pass `set_engine("psvr", a = -1L)` to request odd symmetry instead.

## Usage

``` r
psvr_rmspe_sym_rbf(
  mode = "regression",
  engine = "psvr",
  cost = NULL,
  rbf_sigma = NULL
)

psvr_rmspe_sym_poly(
  mode = "regression",
  engine = "psvr",
  cost = NULL,
  degree = NULL,
  scale_factor = NULL
)

psvr_rmspe_sym_linear(mode = "regression", engine = "psvr", cost = NULL)
```

## Arguments

- mode:

  Only `"regression"` is supported.

- engine:

  Only `"psvr"` is available.

- cost:

  Regularization parameter `Γ > 0`. Use
  [`hardhat::tune()`](https://hardhat.tidymodels.org/reference/tune.html)
  to optimize. Mapped to
  [`cost_psvr()`](https://pbenavidesh.github.io/psvr/reference/cost_psvr.md)
  with range `[-2, 10]` on the log2 scale — wider than
  [`dials::cost()`](https://dials.tidymodels.org/reference/cost.html) to
  cover the larger values needed by LS-SVR models.

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
spec <- psvr_rmspe_sym_rbf(cost = 1000, rbf_sigma = 1) |>
  set_engine("psvr")

spec_poly <- psvr_rmspe_sym_poly(cost = 1000, degree = 2,
                                 scale_factor = 1) |>
  set_engine("psvr")

spec_lin <- psvr_rmspe_sym_linear(cost = 1000) |>
  set_engine("psvr")
} # }
```
