# Summarize a fitted symmetric LS-SVR with RMSPE loss

As
[`summary.psvr_rmspe()`](https://pbenavidesh.github.io/psvr/reference/summary.psvr_rmspe.md),
with the symmetry parameter \\a\\ reported.

## Usage

``` r
# S3 method for class 'psvr_rmspe_sym'
summary(object, ...)
```

## Arguments

- object:

  An object of class `"psvr_rmspe_sym"`, from
  [`psvr_rmspe()`](https://pbenavidesh.github.io/psvr/reference/psvr_rmspe.md)
  with `sym_type = "even"` or `"odd"`.

- ...:

  Ignored.

## Value

`object`, invisibly. Called for the printed summary.

## See also

[`print.psvr_rmspe_sym()`](https://pbenavidesh.github.io/psvr/reference/print.psvr_rmspe_sym.md),
[`coef.psvr_rmspe_sym()`](https://pbenavidesh.github.io/psvr/reference/coef.psvr_rmspe_sym.md)

## Examples

``` r
set.seed(1)
X <- matrix(rnorm(40), 20, 2)
y <- rlnorm(20)
fit <- psvr_rmspe(X, y, sym_type = "even",
                  kernel = make_kernel("rbf", sigma = 1), gamma = 100)
summary(fit)
#> 
#> LS-SVR with RMSPE loss  [psvr_rmspe_sym]
#> 
#>   Kernel:          RBF (sigma = 1)
#>   Training obs.:   20
#>   Predictors:      2
#>   Symmetry:        even  (a = 1)
#> 
#>   Hyperparameters:
#>     Gamma  = 100
#> 
#>   Preconditioner:  applied (diag(1/y) symmetric rescaling)
#> 
```
