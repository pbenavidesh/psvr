# RBF sigma parameter with data-driven range for psvr models

A convenience wrapper that combines
[`sigma_heuristic()`](https://pbenavidesh.github.io/psvr/reference/sigma_heuristic.md)
and
[`rbf_sigma_psvr()`](https://pbenavidesh.github.io/psvr/reference/rbf_sigma_psvr.md)
in a single call. It computes the median pairwise distance from `X` and
returns a `quant_param` whose search range spans one order of magnitude
either side of the heuristic value on the log10 scale (i.e.,
`[log10(sigma_med / width), log10(sigma_med * width)]`).

## Usage

``` r
rbf_sigma_psvr_data(
  X,
  width = 10,
  sample_size = 500L,
  seed = NULL,
  trans = scales::log10_trans()
)
```

## Arguments

- X:

  A numeric matrix or data frame of predictors (already preprocessed —
  centred, scaled, etc.).

- width:

  Positive scalar. Multiplier that sets the half-width of the search
  range around the heuristic sigma. Default `10` (one decade).

- sample_size:

  Integer. Passed to
  [`sigma_heuristic()`](https://pbenavidesh.github.io/psvr/reference/sigma_heuristic.md).
  Default `500L`.

- seed:

  Integer seed for subsampling. Passed to
  [`sigma_heuristic()`](https://pbenavidesh.github.io/psvr/reference/sigma_heuristic.md).
  Default `NULL`.

- trans:

  A `scales` transformation object. Default
  [`scales::log10_trans()`](https://scales.r-lib.org/reference/transform_log.html).

## Value

A `quant_param` dials object with a data-driven search range.

## See also

[`sigma_heuristic()`](https://pbenavidesh.github.io/psvr/reference/sigma_heuristic.md),
[`rbf_sigma_psvr()`](https://pbenavidesh.github.io/psvr/reference/rbf_sigma_psvr.md)

## Examples

``` r
X <- matrix(rnorm(200), ncol = 4)
rbf_sigma_psvr_data(X)
#> RBF Sigma (psvr) (quantitative)
#> Transformer: log-10 [1e-100, Inf]
#> Range (transformed scale): [-0.573, 1.43]
rbf_sigma_psvr_data(X, width = 5)
#> RBF Sigma (psvr) (quantitative)
#> Transformer: log-10 [1e-100, Inf]
#> Range (transformed scale): [-0.272, 1.13]
```
