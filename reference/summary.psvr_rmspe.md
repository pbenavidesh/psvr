# Summarize a fitted LS-SVR with RMSPE loss

Prints the kernel, the training count, the hyperparameter, and whether
the Remark-17 preconditioner fired. No support-vector count is reported:
LS-SVR performs no pruning, so every training point contributes to the
prediction.

## Usage

``` r
# S3 method for class 'psvr_rmspe'
summary(object, ...)
```

## Arguments

- object:

  An object of class `"psvr_rmspe"`, from
  [`psvr_rmspe()`](https://pbenavidesh.github.io/psvr/reference/psvr_rmspe.md)
  with `sym_type = "none"`, or from a parsnip fit unwrapped with
  [`parsnip::extract_fit_engine()`](https://parsnip.tidymodels.org/reference/reexports.html).

- ...:

  Ignored.

## Value

`object`, invisibly. Called for the printed summary.

## See also

[`print.psvr_rmspe()`](https://pbenavidesh.github.io/psvr/reference/print.psvr_rmspe.md),
[`coef.psvr_rmspe()`](https://pbenavidesh.github.io/psvr/reference/coef.psvr_rmspe.md)

## Examples

``` r
set.seed(1)
X <- matrix(rnorm(40), 20, 2)
y <- rlnorm(20)
fit <- psvr_rmspe(X, y, kernel = make_kernel("rbf", sigma = 1), gamma = 100)
summary(fit)
#> 
#> LS-SVR with RMSPE loss  [psvr_rmspe]
#> 
#>   Kernel:          RBF (sigma = 1)
#>   Training obs.:   20
#>   Predictors:      2
#> 
#>   Hyperparameters:
#>     Gamma  = 100
#> 
#>   Preconditioner:  applied (diag(1/y) symmetric rescaling)
#> 
```
