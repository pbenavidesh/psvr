# Fit symmetric LS-SVR with RMSPE loss (Model 4) — internal

Internal fitter for the symmetric RMSPE LS-SVR family. Use
[`psvr()`](https://pbenavidesh.github.io/psvr/reference/psvr.md) with
`loss = "rmspe"` and `sym = +1L` / `-1L` instead. Returns the legacy
`psvr_rmspe_sym` shape, which is also what the parsnip engine fit
wrappers return with `sym_type = "even"` / `"odd"`. The kernel must
satisfy Assumption 3 of the paper (kernel symmetry); see
[`make_kernel()`](https://pbenavidesh.github.io/psvr/reference/make_kernel.md).

## Usage

``` r
.fit_rmspe_sym(X, y, kernel, gamma, a = 1, precondition = "auto")
```

## Arguments

- X, y, kernel, gamma, precondition:

  See [`psvr()`](https://pbenavidesh.github.io/psvr/reference/psvr.md)
  for the full semantics of each argument (including the Remark-17
  preconditioner).

- a:

  Symmetry type: `1` (even) or `-1` (odd). Corresponds to
  `psvr(sym = +1L)` and `psvr(sym = -1L)` respectively;
  [`psvr()`](https://pbenavidesh.github.io/psvr/reference/psvr.md) has
  no `a` argument of its own.

## Value

A list of class `"psvr_rmspe_sym"` (legacy shape).
