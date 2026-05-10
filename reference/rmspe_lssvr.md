# Fit LS-SVR with RMSPE loss (deprecated)

Soft-deprecated wrapper retained for backwards compatibility. New code
should use `psvr(loss = "rmspe", ...)`. Returns the legacy `psvr_rmspe`
object shape.

## Usage

``` r
rmspe_lssvr(X, y, kernel, gamma, precondition = "auto")
```

## Arguments

- X, y:

  Training matrix and strictly-positive target vector.

- kernel:

  Kernel closure from
  [`make_kernel()`](https://pbenavidesh.github.io/psvr/reference/make_kernel.md).

- gamma:

  Regularization parameter `Γ > 0`.

- precondition:

  One of `"auto"`, `"always"`, `"never"`, or a positive numeric
  threshold; controls Remark-17 symmetric rescaling.

## Value

A list of class `"psvr_rmspe"`.

## Examples

``` r
if (FALSE) { # \dontrun{
fit <- psvr(X, y, loss = "rmspe", kernel = make_kernel("rbf"),
            gamma = 100)
} # }
```
