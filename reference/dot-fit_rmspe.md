# Fit LS-SVR with RMSPE loss (Model 3) — internal

Internal fitter for the RMSPE LS-SVR family. Use
[`psvr_rmspe()`](https://pbenavidesh.github.io/psvr/reference/psvr_rmspe.md)
instead. Returns the `psvr_rmspe` shape, which is what
[`psvr_rmspe()`](https://pbenavidesh.github.io/psvr/reference/psvr_rmspe.md)
and the parsnip engine fit wrappers both return.

## Usage

``` r
.fit_rmspe(X, y, kernel, gamma, precondition = "auto")
```

## Arguments

- X, y, kernel, gamma, precondition:

  See
  [`psvr_rmspe()`](https://pbenavidesh.github.io/psvr/reference/psvr_rmspe.md)
  for the full semantics of each argument, including the `diag(1/y)`
  preconditioner that flattens the target-weighted diagonal to a
  constant before the solve.

## Value

A list of class `"psvr_rmspe"`.
