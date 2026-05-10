# Fit symmetric epsilon-SVR with MAPE loss (Model 2) — internal

Internal fitter for the symmetric MAPE epsilon-SVR family. Use
[`psvr()`](https://pbenavidesh.github.io/psvr/reference/psvr.md) with
`loss = "mape"` and `sym = +1L` / `-1L` instead. Returns the legacy
`psvr_mape_sym` shape; the deprecation wrapper
[`mape_sym_svr()`](https://pbenavidesh.github.io/psvr/reference/mape_sym_svr.md)
forwards directly to this function. The kernel must satisfy Assumption 3
of the paper; see
[`make_kernel()`](https://pbenavidesh.github.io/psvr/reference/make_kernel.md).

## Usage

``` r
.fit_mape_sym(
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

- X, y, kernel, C, eps, a, solver, tol:

  See
  [`mape_sym_svr()`](https://pbenavidesh.github.io/psvr/reference/mape_sym_svr.md).

## Value

A list of class `"psvr_mape_sym"` (legacy shape).
