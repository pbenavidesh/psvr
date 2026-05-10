# Fit symmetric epsilon-SVR with MAPE loss (deprecated)

Soft-deprecated wrapper retained for backwards compatibility. New code
should use `psvr(loss = "mape", sym = +1L, ...)` (or `sym = -1L`).
Returns the legacy `psvr_mape_sym` object shape.

## Usage

``` r
mape_sym_svr(
  X,
  y,
  kernel,
  C,
  eps,
  a = 1,
  solver = c("smo", "osqp"),
  tol = 1e-05
)
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

- a:

  Symmetry parameter: `1` (even) or `-1` (odd).

- solver:

  Backend: `"smo"` (default) or `"osqp"`.

- tol:

  Solver zero-threshold.

## Value

A list of class `"psvr_mape_sym"`.

## Examples

``` r
if (FALSE) { # \dontrun{
fit <- psvr(X, y, loss = "mape", sym = +1L,
            kernel = make_kernel("rbf"), C = 10, eps = 5)
} # }
```
