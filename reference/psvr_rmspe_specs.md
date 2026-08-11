# Parsnip model specs: LS-SVR with RMSPE loss (Model 3)

Create parsnip model specifications for
[`psvr_rmspe()`](https://pbenavidesh.github.io/psvr/reference/psvr_rmspe.md)
with a fixed kernel type. `cost` maps to the regularization parameter
\\\Gamma\\.

## Usage

``` r
psvr_rmspe_rbf(
  mode = "regression",
  engine = "psvr",
  cost = NULL,
  rbf_sigma = NULL,
  sym_type = NULL
)

psvr_rmspe_poly(
  mode = "regression",
  engine = "psvr",
  cost = NULL,
  degree = NULL,
  scale_factor = NULL,
  sym_type = NULL
)

psvr_rmspe_linear(
  mode = "regression",
  engine = "psvr",
  cost = NULL,
  sym_type = NULL
)
```

## Arguments

- mode:

  Only `"regression"` is supported.

- engine:

  Only `"psvr"` is available.

- cost:

  Regularization parameter \\\Gamma \> 0\\. Use
  [`hardhat::tune()`](https://hardhat.tidymodels.org/reference/tune.html)
  to optimize. Mapped to
  [`cost_psvr()`](https://pbenavidesh.github.io/psvr/reference/cost_psvr.md),
  whose default range `[-2, 10]` on the log2 scale (\\\Gamma \le 1024\\)
  is the \\\epsilon\\-SVR range and is **too narrow for LS-SVR**.
  \\\Gamma\\ enters the LS-SVR system only through the \\y_k^2/\Gamma\\
  diagonal, so the value that balances that term against the kernel
  scales with the outcome's variance and with the sample size — the
  quantity
  [`cost_psvr_ls_data()`](https://pbenavidesh.github.io/psvr/reference/cost_psvr_ls_data.md)
  computes. The LS-SVR optimum is therefore routinely orders of
  magnitude above the static ceiling, and a grid over the default is
  boundary-trapped whenever it is. Pass
  [`cost_psvr_ls_data()`](https://pbenavidesh.github.io/psvr/reference/cost_psvr_ls_data.md)
  built from the training outcome explicitly, via
  [`update()`](https://rdrr.io/r/stats/update.html) on the extracted
  parameter set or via
  [`psvr_option_add_cost_ls()`](https://pbenavidesh.github.io/psvr/reference/psvr_option_add_cost_ls.md)
  for a workflow set. This **cannot** be automated: `tune` finalizes
  parameters from the molded **predictors** only and never passes the
  outcome to
  [`dials::finalize()`](https://dials.tidymodels.org/reference/finalize.html),
  so no `finalize` function on `cost` could compute it.

- rbf_sigma:

  RBF bandwidth \\\sigma \> 0\\. Use
  [`hardhat::tune()`](https://hardhat.tidymodels.org/reference/tune.html)
  to optimize. Mapped to
  [`rbf_sigma_psvr()`](https://pbenavidesh.github.io/psvr/reference/rbf_sigma_psvr.md),
  whose default range `[-3, 1]` on the log10 scale is a fixed,
  conservative fallback. **The range does not finalize automatically
  from the training data.**
  [`rbf_sigma_psvr()`](https://pbenavidesh.github.io/psvr/reference/rbf_sigma_psvr.md)
  sets `finalize = NULL`, so
  [`dials::finalize()`](https://dials.tidymodels.org/reference/finalize.html)
  leaves it untouched. To centre the range on the data, pass
  [`rbf_sigma_psvr_data()`](https://pbenavidesh.github.io/psvr/reference/rbf_sigma_psvr_data.md)
  computed on the **preprocessed** predictors explicitly — via
  [`update()`](https://rdrr.io/r/stats/update.html) on the extracted
  parameter set, or via
  [`psvr_option_add()`](https://pbenavidesh.github.io/psvr/reference/psvr_option_add.md)
  for a workflow set. (RBF specs only.)

- sym_type:

  Symmetry type: `"none"` (default) fits the non-symmetric LS-SVR of
  Model 3; `"even"` (a = 1) and `"odd"` (a = -1) fit the symmetric
  LS-SVR of Model 4. Use
  [`hardhat::tune()`](https://hardhat.tidymodels.org/reference/tune.html)
  to optimise over the levels during CV; see
  [`sym_type_param()`](https://pbenavidesh.github.io/psvr/reference/sym_type_param.md)
  to restrict which levels are searched.

- degree:

  Polynomial degree \\\ge 1\\. Use
  [`hardhat::tune()`](https://hardhat.tidymodels.org/reference/tune.html)
  to optimize. (Polynomial specs only.)

- scale_factor:

  Polynomial constant term (`coef0`). Use
  [`hardhat::tune()`](https://hardhat.tidymodels.org/reference/tune.html)
  to optimize. (Polynomial specs only.)

## Value

A parsnip `model_spec` object of the corresponding class.

## Engine arguments

The `precondition` argument of
[`psvr_rmspe()`](https://pbenavidesh.github.io/psvr/reference/psvr_rmspe.md)
is exposed as a non-tunable engine argument. Pass it via
[`parsnip::set_engine()`](https://parsnip.tidymodels.org/reference/set_engine.html),
e.g. `set_engine("psvr", precondition = "always")`. Default is `"auto"`.
See
[`psvr_rmspe()`](https://pbenavidesh.github.io/psvr/reference/psvr_rmspe.md)
for accepted values and semantics.

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

# Symmetric LS-SVR (Model 4) via the sym_type argument:
spec_sym <- psvr_rmspe_rbf(cost = 1000, rbf_sigma = 1,
                           sym_type = "even") |>
  set_engine("psvr")
} # }
```
