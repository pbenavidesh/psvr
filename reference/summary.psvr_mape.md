# Summarize a fitted epsilon-SVR with MAPE loss

Prints the kernel, the training and support-vector counts, the
hyperparameters, and the SMO iteration count with its convergence
status. Every training point contributes for LS-SVR but not here: the
support-vector percentage is the sparsity of the fit.

## Usage

``` r
# S3 method for class 'psvr_mape'
summary(object, ...)
```

## Arguments

- object:

  An object of class `"psvr_mape"`, from
  [`psvr_mape()`](https://pbenavidesh.github.io/psvr/reference/psvr_mape.md)
  with `sym_type = "none"`, or from a parsnip fit unwrapped with
  [`parsnip::extract_fit_engine()`](https://parsnip.tidymodels.org/reference/reexports.html).

- ...:

  Ignored.

## Value

`object`, invisibly. Called for the printed summary.

## See also

[`print.psvr_mape()`](https://pbenavidesh.github.io/psvr/reference/print.psvr_mape.md),
[`coef.psvr_mape()`](https://pbenavidesh.github.io/psvr/reference/coef.psvr_mape.md)

## Examples

``` r
set.seed(1)
X <- matrix(rnorm(40), 20, 2)
y <- rlnorm(20)
fit <- psvr_mape(X, y, kernel = make_kernel("rbf", sigma = 1),
                 C = 10, eps = 5)
summary(fit)
#> 
#> Epsilon-SVR with MAPE loss  [psvr_mape]
#> 
#>   Kernel:          RBF (sigma = 1)
#>   Training obs.:   20
#>   Predictors:      2
#>   Support vectors: 20 (100.0%)
#> 
#>   Hyperparameters:
#>     C      = 10
#>     eps    = 5
#> 
#>   SMO iterations:  3039 (converged)
#> 
```
