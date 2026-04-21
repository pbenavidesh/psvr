# Cost parameter with extended range for psvr models

A dials parameter for the regularisation parameter in psvr models. The
default range \[-2, 10\] on the log2 scale (corresponding to
approximately 0.25 to 1024) is wider than
[`dials::cost()`](https://dials.tidymodels.org/reference/cost.html) to
accommodate the larger regularisation values typically needed by LS-SVR
models.

## Usage

``` r
cost_psvr(range = c(-2, 10), trans = scales::log2_trans())
```

## Arguments

- range:

  Numeric vector of length 2 on the log2 scale. Default `c(-2, 10)`.

- trans:

  A `scales` transformation object. Default
  [`scales::log2_trans()`](https://scales.r-lib.org/reference/transform_log.html).

## Value

A `quant_param` dials object.

## Examples

``` r
cost_psvr()
#> Cost (quantitative)
#> Transformer: log-2 [1e-100, Inf]
#> Range (transformed scale): [-2, 10]
```
