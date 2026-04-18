# Parsnip model spec: symmetric epsilon-SVR with MAPE loss (Model 2)

Creates a parsnip model specification for
[`mape_sym_svr()`](https://pbenavidesh.github.io/psvr/reference/mape_sym_svr.md).
The kernel, symmetry parameter `a`, and solver tolerance are engine
arguments.

## Usage

``` r
psvr_mape_sym(
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

  Regularization parameter `C > 0`.

- svm_margin:

  Epsilon tube half-width `ε ≥ 0` (percentage units).

## Value

A `psvr_mape_sym_model` / `model_spec` object.

## Examples

``` r
if (FALSE) { # \dontrun{
library(parsnip)
K <- make_kernel("rbf", sigma = 1)
spec <- psvr_mape_sym(cost = 10, svm_margin = 5) |>
  set_engine("psvr", kernel = K, a = 1L)
fit(spec, mpg ~ ., data = mtcars)
} # }
```
