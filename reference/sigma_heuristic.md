# Median-distance heuristic for RBF kernel bandwidth

Returns the median pairwise Euclidean distance between rows of `X`,
which is a standard data-driven starting point for the RBF kernel
bandwidth (Schölkopf & Smola, 2002). Use the result to define a sensible
`rbf_sigma` search range centred on this value via
`dials::rbf_sigma(range = c(log10(sigma / 10), log10(sigma * 10)))`.

## Usage

``` r
sigma_heuristic(X, sample_size = 500L, seed = NULL)
```

## Arguments

- X:

  A numeric matrix or data frame of predictors (already preprocessed —
  centred, scaled, etc.).

- sample_size:

  Integer. If `nrow(X) > sample_size`, a random subsample is used to
  avoid O(n²) memory cost on large datasets. Default `500L`.

- seed:

  Integer seed for the subsample. Default `NULL`.

## Value

A scalar numeric: the median pairwise Euclidean distance.

## Examples

``` r
X <- matrix(rnorm(200), ncol = 4)
sigma_heuristic(X)
#> [1] 2.673608
```
