# Cross-validate psvr() with automatic warm-start across folds

Fits a `psvr(loss = "mape")` model on each split in `splits`, carrying
the converged `(alpha, alpha_star)` from one fold into the next as the
SMO warm-start (Theorem 5 of arXiv:2605.01446 v3, Algorithm 1). Folds
are projected to feasibility before each solve. Returns a tibble with
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
  [`psvr()`](https://pbenavidesh.github.io/psvr/reference/psvr.md). Must
  specify `kernel` and the MAPE hyperparameters (`C`, `eps`).
  `alpha_init` and `alpha_star_init` are managed internally; supplying
  them via `...` is an error.

- X_var:

  Character vector of predictor column names.

- y_var:

  Single character giving the target column name.

- warm_start:

  Logical; if `FALSE`, each fold fits cold-start (useful for
  benchmarking the T5 speedup).

- verbose:

  Logical; if `TRUE`, print per-fold progress.

## Value

A `tibble` with one row per split and columns:

- `split_id`:

  1-based fold index.

- `fit`:

  A list-column of `psvr_fit` objects.

- `predictions`:

  A list-column of numeric vectors (predictions on the assessment set).

- `metrics`:

  A list-column of named numeric vectors (`mape`, `rmspe`, `mse`, `r2`).

- `iter_count`:

  Integer; SMO iterations from `fit$solver_meta$iters`.

- `elapsed_sec`:

  Numeric; wall-clock seconds for the fit.

- `warm_started`:

  Logical; `TRUE` for fold \> 1 when `warm_start = TRUE`.

## Details

This helper currently only supports `loss = "mape"`. For
`loss = "rmspe"` (LS-SVR), each fold is a single linear-system solve
with no carryover state; use
[`tune::tune_grid()`](https://tune.tidymodels.org/reference/tune_grid.html)
with parallel cold-start instead.

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
                 loss = "mape",
                 kernel = make_kernel("rbf", sigma = 1),
                 C = 10, eps = 5)
  median(vapply(res$metrics, function(m) m[["mape"]], numeric(1)))
}
#> Warning: SMO solver did not converge within max_iter = 100000 (final iter = 100000)
#> [1] 157.6322
```
