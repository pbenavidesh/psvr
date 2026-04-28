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

## Details

For LS-SVR (`m3`, `m4`) models, the static range may still be too narrow
on benchmark datasets where the optimum exceeds 10⁴. Prefer
[`cost_psvr_ls_data()`](https://pbenavidesh.github.io/psvr/reference/cost_psvr_ls_data.md)
in that case.

## See also

[`cost_psvr_ls_data()`](https://pbenavidesh.github.io/psvr/reference/cost_psvr_ls_data.md)
for a data-driven LS-SVR variant.

## Examples

``` r
cost_psvr()
#> Cost (quantitative)
#> Transformer: log-2 [1e-100, Inf]
#> Range (transformed scale): [-2, 10]
```
