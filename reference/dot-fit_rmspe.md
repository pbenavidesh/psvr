# Fit LS-SVR with RMSPE loss (Model 3) — internal

Internal fitter for the RMSPE LS-SVR family. Use
[`psvr()`](https://pbenavidesh.github.io/psvr/reference/psvr.md) with
`loss = "rmspe"` instead. Returns the legacy `psvr_rmspe` shape; the
deprecation wrapper
[`rmspe_lssvr()`](https://pbenavidesh.github.io/psvr/reference/rmspe_lssvr.md)
forwards directly to this function.

## Usage

``` r
.fit_rmspe(X, y, kernel, gamma, precondition = "auto")
```

## Arguments

- X, y, kernel, gamma, precondition:

  See
  [`rmspe_lssvr()`](https://pbenavidesh.github.io/psvr/reference/rmspe_lssvr.md)
  for the full semantics of each argument (including the Remark-17
  preconditioner).

## Value

A list of class `"psvr_rmspe"` (legacy shape).
