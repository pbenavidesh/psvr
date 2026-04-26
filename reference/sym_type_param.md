# Dials parameter for symmetry type

Returns a qualitative
[`dials::new_qual_param()`](https://dials.tidymodels.org/reference/new-param.html)
describing the `sym_type` argument of symmetric psvr model specs.
`"even"` maps to `a = 1L` (standard symmetric kernel); `"odd"` maps to
`a = -1L` (anti-symmetric kernel).

## Usage

``` r
sym_type_param()
```

## Value

A `qual_param` object.
