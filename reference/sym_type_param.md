# Dials parameter for symmetry type

Returns a qualitative
[`dials::new_qual_param()`](https://dials.tidymodels.org/reference/new-param.html)
describing the `sym_type` argument of the psvr model specs. `"none"`
fits the non-symmetric model; `"even"` maps to `a = 1L` (standard
symmetric kernel); `"odd"` maps to `a = -1L` (anti-symmetric kernel).

## Usage

``` r
sym_type_param(values = c("none", "even", "odd"))
```

## Arguments

- values:

  Character vector of levels to search over. Any subset of
  `c("none", "even", "odd")`; defaults to all three. Pass
  `values = c("even", "odd")` to tune over the symmetric models only,
  which reproduces the two-level grid offered before psvr 0.0.2.9011.

## Value

A `qual_param` object.

## Examples

``` r
sym_type_param()
#> Symmetry type (qualitative)
#> 3 possible values include:
#> 'none', 'even', and 'odd'
sym_type_param(values = c("even", "odd"))
#> Symmetry type (qualitative)
#> 2 possible values include:
#> 'even' and 'odd'
```
