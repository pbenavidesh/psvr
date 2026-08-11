# Data-driven cost range for LS-SVR psvr models

Returns a `quant_param` whose search range scales with `var(y) * N`, the
standard heuristic for the LS-SVR regularisation parameter \\\Gamma\\
(Suykens et al. 2002, *Least Squares Support Vector Machines*, §3.1.3).
On the log2 scale, the lower bound is `-2` (i.e. \\\Gamma \ge 0.25\\)
and the upper bound is `log2(var(y) * N) + width_log2`.

## Usage

``` r
cost_psvr_ls_data(y, n = length(y), width_log2 = 4)
```

## Arguments

- y:

  Numeric vector of strictly positive training targets.

- n:

  Sample size. Default `length(y)`.

- width_log2:

  Scalar giving the half-width (in log2 units) added above
  `log2(var(y) * n)` to set the upper bound. Default `4` (about 16×
  headroom). Negative values are accepted with a warning, since the
  resulting upper bound falls below the `var(y) * n` heuristic and is
  unlikely to be useful.

## Value

A `quant_param` dials object.

## Details

The default `width_log2 = 4` places the upper bound about 16 times above
`var(y) * n`, which is headroom for a Bayesian or grid search to work in
without pinning against the boundary. Because the bound tracks
`var(y) * n`, it moves with the outcome: the static
[`cost_psvr()`](https://pbenavidesh.github.io/psvr/reference/cost_psvr.md)
range \[-2, 10\] (i.e. \\\Gamma \le 1024\\) is fixed, so it falls short
whenever `var(y) * n` exceeds a few hundred — which is the usual case,
not the exception.

Use this function for `m3` (LS-SVR) and `m4` (symmetric LS-SVR)
workflows. Stick to
[`cost_psvr()`](https://pbenavidesh.github.io/psvr/reference/cost_psvr.md)
for `m1`/`m2` (\\\epsilon\\-SVR), where `cost` maps to `C` and typical
optima lie in \[10, 100\].

## See also

[`cost_psvr()`](https://pbenavidesh.github.io/psvr/reference/cost_psvr.md),
[`psvr_option_add_cost_ls()`](https://pbenavidesh.github.io/psvr/reference/psvr_option_add_cost_ls.md)

## Examples

``` r
cost_psvr_ls_data(c(10, 20, 30, 40, 50))
#> Cost (quantitative)
#> Transformer: log-2 [1e-100, Inf]
#> Range (transformed scale): [-2, 14.3]
```
