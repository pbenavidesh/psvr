# Data-driven cost range for LS-SVR psvr models

Returns a `quant_param` whose search range scales with `var(y) * N`, the
standard heuristic for the LS-SVR regularisation parameter `Γ` (Suykens
et al. 2002, *Least Squares Support Vector Machines*, §3.1.3). On the
log2 scale, the lower bound is `-2` (i.e. `Γ ≥ 0.25`) and the upper
bound is `log2(var(y) * N) + width_log2`.

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
  `log2(var(y) * n)` to set the upper bound. Default `4` (≈16×
  headroom). Negative values are accepted with a warning, since the
  resulting upper bound falls below the `var(y) * n` heuristic and is
  unlikely to be useful.

## Value

A `quant_param` dials object.

## Details

With the default `width_log2 = 4`, the upper bound covers the typical
optimum within ~2 orders of magnitude on benchmark datasets. For Boston
Housing (`var(medv) ≈ 84.6`, `N_train = 404` under an 80/20 split), this
gives an upper bound of `2^19.06 ≈ 5.4 × 10⁵`, two decades above the
published optimum `Γ ≈ 1.7 × 10⁴` — comfortable headroom for Bayesian
optimisation without boundary trapping. The static
[`cost_psvr()`](https://pbenavidesh.github.io/psvr/reference/cost_psvr.md)
range \[-2, 10\] (i.e. `Γ ≤ 1024`) underestimates this by more than a
decade.

Use this function for `m3` (LS-SVR) and `m4` (symmetric LS-SVR)
workflows. Stick to
[`cost_psvr()`](https://pbenavidesh.github.io/psvr/reference/cost_psvr.md)
for `m1`/`m2` (`ε`-SVR), where `cost` maps to `C` and typical optima lie
in \[10, 100\].

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
