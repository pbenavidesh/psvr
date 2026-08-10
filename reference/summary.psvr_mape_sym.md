# Summarize a fitted symmetric epsilon-SVR with MAPE loss

As
[`summary.psvr_mape()`](https://pbenavidesh.github.io/psvr/reference/summary.psvr_mape.md),
with the symmetry parameter \\a\\ reported.

## Usage

``` r
# S3 method for class 'psvr_mape_sym'
summary(object, ...)
```

## Arguments

- object:

  An object of class `"psvr_mape_sym"`, from
  [`psvr_mape()`](https://pbenavidesh.github.io/psvr/reference/psvr_mape.md)
  with `sym_type = "even"` or `"odd"`.

- ...:

  Ignored.

## Value

`object`, invisibly. Called for the printed summary.

## See also

[`print.psvr_mape_sym()`](https://pbenavidesh.github.io/psvr/reference/print.psvr_mape_sym.md),
[`coef.psvr_mape_sym()`](https://pbenavidesh.github.io/psvr/reference/coef.psvr_mape_sym.md)

## Examples

``` r
set.seed(1)
X <- matrix(rnorm(40), 20, 2)
y <- rlnorm(20)
fit <- psvr_mape(X, y, sym_type = "even",
                 kernel = make_kernel("rbf", sigma = 1), C = 10, eps = 5)
summary(fit)
#> 
#> Epsilon-SVR with MAPE loss  [psvr_mape_sym]
#> 
#>   Kernel:          RBF (sigma = 1)
#>   Training obs.:   20
#>   Predictors:      2
#>   Support vectors: 19 (95.0%)
#>   Symmetry:        even  (a = 1)
#> 
#>   Hyperparameters:
#>     C      = 10
#>     eps    = 5
#> 
#>   SMO iterations:  2926 (converged)
#> 
```
