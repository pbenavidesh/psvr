# Fit epsilon-SVR with MAPE loss (deprecated)

Soft-deprecated wrapper retained for backwards compatibility. New code
should use `psvr(loss = "mape", ...)`. Returns the legacy `psvr_mape`
object shape; the legacy
[`predict.psvr_mape()`](https://pbenavidesh.github.io/psvr/reference/predict.psvr_mape.md)
/
[`print.psvr_mape()`](https://pbenavidesh.github.io/psvr/reference/print.psvr_mape.md)
/
[`coef.psvr_mape()`](https://pbenavidesh.github.io/psvr/reference/coef.psvr_mape.md)
methods continue to dispatch correctly.

## Usage

``` r
mape_svr(X, y, kernel, C, eps, solver = c("smo", "osqp"), tol = 1e-05)
```

## Arguments

- X, y:

  Training matrix and strictly-positive target vector.

- kernel:

  Kernel closure from
  [`make_kernel()`](https://pbenavidesh.github.io/psvr/reference/make_kernel.md).

- C:

  Regularization parameter `C > 0`.

- eps:

  Insensitivity tube half-width (% units), `eps >= 0`.

- solver:

  Backend: `"smo"` (default) or `"osqp"`.

- tol:

  Solver zero-threshold.

## Value

A list of class `"psvr_mape"`.

## Examples

``` r
if (FALSE) { # \dontrun{
# Use psvr() instead:
fit <- psvr(X, y, loss = "mape", kernel = make_kernel("rbf"),
            C = 10, eps = 5)
} # }
```
