# Fit symmetric LS-SVR with RMSPE loss (Model 4) — internal

Internal fitter for the symmetric RMSPE LS-SVR family. Use
[`psvr()`](https://pbenavidesh.github.io/psvr/reference/psvr.md) with
`loss = "rmspe"` and `sym = +1L` / `-1L` instead. Returns the legacy
`psvr_rmspe_sym` shape; the deprecation wrapper
[`rmspe_sym_lssvr()`](https://pbenavidesh.github.io/psvr/reference/rmspe_sym_lssvr.md)
forwards directly to this function. The kernel must satisfy Assumption 3
of the paper (kernel symmetry); see
[`make_kernel()`](https://pbenavidesh.github.io/psvr/reference/make_kernel.md).

## Usage

``` r
.fit_rmspe_sym(X, y, kernel, gamma, a = 1, precondition = "auto")
```

## Arguments

- X, y, kernel, gamma, a, precondition:

  See
  [`rmspe_sym_lssvr()`](https://pbenavidesh.github.io/psvr/reference/rmspe_sym_lssvr.md)
  for the full semantics of each argument (including the Remark-17
  preconditioner).

## Value

A list of class `"psvr_rmspe_sym"` (legacy shape).
