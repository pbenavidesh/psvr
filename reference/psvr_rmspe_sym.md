# Parsnip model spec: symmetric LS-SVR with RMSPE loss (Model 4)

Creates a parsnip model specification for
[`rmspe_sym_lssvr()`](https://pbenavidesh.github.io/psvr/reference/rmspe_sym_lssvr.md).
The kernel and symmetry parameter `a` are engine arguments.

## Usage

``` r
psvr_rmspe_sym(mode = "regression", engine = "psvr", cost = NULL)
```

## Arguments

- mode:

  Only `"regression"` is supported.

- engine:

  Only `"psvr"` is available.

- cost:

  Regularization parameter `Γ > 0`.

## Value

A `psvr_rmspe_sym_model` / `model_spec` object.

## Examples

``` r
if (FALSE) { # \dontrun{
library(parsnip)
K <- make_kernel("rbf", sigma = 1)
spec <- psvr_rmspe_sym(cost = 1) |>
  set_engine("psvr", kernel = K, a = 1L)
fit(spec, mpg ~ ., data = mtcars)
} # }
```
