# Parsnip model specs: epsilon-SVR with MAPE loss (Model 1)

Create parsnip model specifications for
[`psvr_mape()`](https://pbenavidesh.github.io/psvr/reference/psvr_mape.md)
with a fixed kernel type. Kernel parameters are tunable parsnip
arguments; the symmetry parameter `a` and solver tolerance are engine
arguments passed via `set_engine()`.

## Usage

``` r
psvr_mape_rbf(
  mode = "regression",
  engine = "psvr",
  cost = NULL,
  margin = NULL,
  rbf_sigma = NULL,
  sym_type = NULL
)

psvr_mape_poly(
  mode = "regression",
  engine = "psvr",
  cost = NULL,
  margin = NULL,
  degree = NULL,
  scale_factor = NULL,
  sym_type = NULL
)

psvr_mape_linear(
  mode = "regression",
  engine = "psvr",
  cost = NULL,
  margin = NULL,
  sym_type = NULL
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
  [`cost_psvr()`](https://pbenavidesh.github.io/psvr/reference/cost_psvr.md),
  whose default range is `[-2, 10]` on the log2 scale (about 0.25 to
  1024). That range is adequate here, where typical \\\epsilon\\-SVR
  optima lie in \[10, 100\]. The LS-SVR specs
  ([psvr_rmspe_specs](https://pbenavidesh.github.io/psvr/reference/psvr_rmspe_specs.md))
  map `cost` to \\\Gamma\\ and need a much wider range; see there.

- margin:

  Epsilon tube half-width \\\epsilon \ge 0\\ expressed as a percentage
  of each target value. Use
  [`hardhat::tune()`](https://hardhat.tidymodels.org/reference/tune.html)
  to optimize. Mapped to
  [`margin_percentage()`](https://pbenavidesh.github.io/psvr/reference/margin_percentage.md)
  with default range `[1, 20]` (percentage units). Named to match
  [`parsnip::svm_rbf()`](https://parsnip.tidymodels.org/reference/svm_rbf.html);
  note the units differ from
  [`dials::svm_margin()`](https://dials.tidymodels.org/reference/cost.html),
  which is absolute rather than percentage.

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

  Symmetry type: `"none"` (default) fits the non-symmetric
  \\\epsilon\\-SVR of Model 1; `"even"` (a = 1) and `"odd"` (a = -1) fit
  the symmetric \\\epsilon\\-SVR of Model 2. Use
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

## Examples

``` r
if (FALSE) { # \dontrun{
library(parsnip)
spec <- psvr_mape_rbf(cost = 10, margin = 1, rbf_sigma = 1) |>
  set_engine("psvr")

spec_poly <- psvr_mape_poly(cost = 10, margin = 1, degree = 2,
                            scale_factor = 1) |>
  set_engine("psvr")

spec_lin <- psvr_mape_linear(cost = 10, margin = 1) |>
  set_engine("psvr")

# Symmetric epsilon-SVR (Model 2) via the sym_type argument:
spec_sym <- psvr_mape_rbf(cost = 10, margin = 1, rbf_sigma = 1,
                          sym_type = "even") |>
  set_engine("psvr")
} # }
```
