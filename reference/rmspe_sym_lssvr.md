# Fit symmetric LS-SVR with RMSPE loss (deprecated)

Soft-deprecated wrapper retained for backwards compatibility. New code
should use `psvr(loss = "rmspe", sym = +1L, ...)` (or `sym = -1L`).
Returns the legacy `psvr_rmspe_sym` object shape.

## Usage

``` r
rmspe_sym_lssvr(X, y, kernel, gamma, a = 1, precondition = "auto")
```

## Arguments

- X, y:

  Training matrix and strictly-positive target vector.

- kernel:

  Kernel closure from
  [`make_kernel()`](https://pbenavidesh.github.io/psvr/reference/make_kernel.md).

- gamma:

  Regularization parameter `Γ > 0`.

- a:

  Symmetry parameter: `1` (even) or `-1` (odd).

- precondition:

  One of `"auto"`, `"always"`, `"never"`, or a positive numeric
  threshold; controls Remark-17 symmetric rescaling.

## Value

A list of class `"psvr_rmspe_sym"`.

## Examples

``` r
if (FALSE) { # \dontrun{
fit <- psvr(X, y, loss = "rmspe", sym = +1L,
            kernel = make_kernel("rbf"), gamma = 100)
} # }
```
