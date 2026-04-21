# RBF sigma parameter with median-distance finalization

A dials parameter for the RBF kernel bandwidth in psvr models. When used
inside `tune_grid()` or `workflow_map()`, the `finalize` function
automatically sets the search range using the median-distance heuristic
computed from the preprocessed training data.

## Usage

``` r
rbf_sigma_psvr(range = c(-3, 1), trans = scales::log10_trans())
```

## Arguments

- range:

  Numeric vector of length 2 on the log10 scale. Used as fallback if no
  training data are available. Default `c(-3, 1)`.

- trans:

  A `scales` transformation object. Default
  [`scales::log10_trans()`](https://scales.r-lib.org/reference/transform_log.html).

## Value

A `quant_param` dials object with finalize support.

## Examples

``` r
rbf_sigma_psvr()
#> RBF Sigma (psvr) (quantitative)
#> Transformer: log-10 [1e-100, Inf]
#> Range (transformed scale): [-3, 1]
```
