# Insensitivity margin in percentage units

A dials parameter for the epsilon tube half-width in psvr MAPE models.
Unlike
[`dials::svm_margin()`](https://dials.tidymodels.org/reference/cost.html)
which uses absolute units, this parameter is expressed as a percentage
of each target value. The default range \[1, 20\] means the
insensitivity tube spans 1% to 20% of each target.

## Usage

``` r
margin_percentage(range = c(1, 20), trans = NULL)
```

## Arguments

- range:

  Numeric vector of length 2. Default `c(1, 20)`.

- trans:

  A `scales` transformation object. Default `NULL`.

## Value

A `quant_param` dials object.

## Examples

``` r
margin_percentage()
#> Insensitivity Margin (%) (quantitative)
#> Range: [1, 20]
margin_percentage(range = c(0.5, 10))
#> Insensitivity Margin (%) (quantitative)
#> Range: [0.5, 10]
```
