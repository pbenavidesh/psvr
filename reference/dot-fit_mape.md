# Fit epsilon-SVR with MAPE loss (Model 1) — internal

Internal fitter for the MAPE epsilon-SVR family. Use
[`psvr()`](https://pbenavidesh.github.io/psvr/reference/psvr.md) with
`loss = "mape"` instead. Returns the legacy `psvr_mape` shape; the
deprecation wrapper
[`mape_svr()`](https://pbenavidesh.github.io/psvr/reference/mape_svr.md)
forwards directly to this function.

## Usage

``` r
.fit_mape(X, y, kernel, C, eps, solver = c("smo", "osqp"), tol = 1e-05)
```

## Arguments

- X, y, kernel, C, eps, solver, tol:

  See
  [`mape_svr()`](https://pbenavidesh.github.io/psvr/reference/mape_svr.md).

## Value

A list of class `"psvr_mape"` (legacy shape).
