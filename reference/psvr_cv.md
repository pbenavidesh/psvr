# Cross-validate psvr_mape() with automatic warm-start across folds

Fits a
[`psvr_mape()`](https://pbenavidesh.github.io/psvr/reference/psvr_mape.md)
model on each split in `splits`, carrying the converged dual variables
`(alpha, alpha_star)` from one fold into the next as the SMO warm-start,
so each solve starts from the previous fold's optimum instead of from
zero. Because consecutive folds share most of their training rows, that
starting point is already close to feasible; before each solve the
carried vectors are projected back onto the constraint set (the equality
\\\sum_k \beta_k = 0\\ and the per-sample box), with the residual
violation absorbed by the rows that are new to this fold. The warm-start
procedure is Algorithm 1 of arXiv:2605.01446 v3. Returns a tibble with
one row per fold.

## Usage

``` r
psvr_cv(
  splits,
  ...,
  X_var = NULL,
  y_var = NULL,
  warm_start = TRUE,
  verbose = FALSE
)
```

## Arguments

- splits:

  Either an `rsample::rset` object (e.g. from
  [`rsample::vfold_cv()`](https://rsample.tidymodels.org/reference/vfold_cv.html)),
  or a list of named lists each containing `analysis` (data frame),
  `assessment` (data frame), and optionally `row_ids` (integer vector of
  original training-row indices used for warm-start alignment across
  folds; defaults to positional).

- ...:

  Arguments forwarded to
  [`psvr_mape()`](https://pbenavidesh.github.io/psvr/reference/psvr_mape.md).
  Must specify `kernel` and the MAPE hyperparameters (`C`, `eps`).
  `alpha_init` and `alpha_star_init` are managed internally; supplying
  them via `...` is an error, and so is `loss`, which is not an argument
  of
  [`psvr_mape()`](https://pbenavidesh.github.io/psvr/reference/psvr_mape.md).

- X_var:

  Character vector of predictor column names.

- y_var:

  Single character giving the target column name.

- warm_start:

  Logical; if `FALSE`, each fold fits cold-start (useful for
  benchmarking the T5 speedup).

- verbose:

  Logical; if `TRUE`, report per-fold progress via
  [`message()`](https://rdrr.io/r/base/message.html) (suppressible with
  [`suppressMessages()`](https://rdrr.io/r/base/message.html)).

## Value

A `tibble` with one row per split and columns:

- `split_id`:

  1-based fold index.

- `fit`:

  A list-column of `psvr_mape` objects, or `psvr_mape_sym` when
  `sym_type` is `"even"` or `"odd"`.

- `predictions`:

  A list-column of numeric vectors (predictions on the assessment set).

- `metrics`:

  A list-column of named numeric vectors (`mape`, `rmspe`, `mse`, `r2`).

- `iter_count`:

  Integer; SMO iterations from `fit$iterations`.

- `elapsed_sec`:

  Numeric; wall-clock seconds for the fit.

- `warm_started`:

  Logical; `TRUE` for fold \> 1 when `warm_start = TRUE`.

## Details

This helper is **MAPE-only**, and there is no `loss` argument. That is a
limitation of the implementation, not of the method: only
[`psvr_mape()`](https://pbenavidesh.github.io/psvr/reference/psvr_mape.md)
was ever wired to it. LS-SVR cross-validates perfectly well, it simply
has no carryover state to exploit (each fold is a single linear-system
solve), so for
[`psvr_rmspe()`](https://pbenavidesh.github.io/psvr/reference/psvr_rmspe.md)
use
[`tune::tune_grid()`](https://tune.tidymodels.org/reference/tune_grid.html)
with parallel cold-start.

## Examples

``` r
if (requireNamespace("rsample", quietly = TRUE) &&
    requireNamespace("tibble",  quietly = TRUE)) {
  set.seed(2026)
  d <- data.frame(
    y  = stats::rlnorm(80, sdlog = 1.0),
    x1 = stats::rnorm(80),
    x2 = stats::rnorm(80)
  )
  folds <- rsample::vfold_cv(d, v = 5)
  res <- psvr_cv(folds, X_var = c("x1", "x2"), y_var = "y",
                 kernel = make_kernel("rbf", sigma = 1),
                 C = 10, eps = 5)
  median(vapply(res$metrics, function(m) m[["mape"]], numeric(1)))
}
#> Warning: SMO solver did not converge within max_iter = 100000 (final iter = 100000)
#> [1] 157.6322
```
