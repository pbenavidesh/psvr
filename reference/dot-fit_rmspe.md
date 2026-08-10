# Fit LS-SVR with RMSPE loss (Model 3) — internal

Internal fitter for the RMSPE LS-SVR family. Use
[`psvr()`](https://pbenavidesh.github.io/psvr/reference/psvr.md) with
`loss = "rmspe"` instead. Returns the legacy `psvr_rmspe` shape, which
is also what the parsnip engine fit wrappers return.

## Usage

``` r
.fit_rmspe(X, y, kernel, gamma, precondition = "auto")
```

## Arguments

- X, y, kernel, gamma, precondition:

  See [`psvr()`](https://pbenavidesh.github.io/psvr/reference/psvr.md)
  for the full semantics of each argument (including the Remark-17
  preconditioner).

## Value

A list of class `"psvr_rmspe"` (legacy shape).
