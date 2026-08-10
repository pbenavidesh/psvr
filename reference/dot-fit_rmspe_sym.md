# Fit symmetric LS-SVR with RMSPE loss (Model 4) — internal

Internal fitter for the symmetric RMSPE LS-SVR family. Use
[`psvr_rmspe()`](https://pbenavidesh.github.io/psvr/reference/psvr_rmspe.md)
with `sym_type = "even"` / `"odd"` instead. Returns the `psvr_rmspe_sym`
shape, which is what
[`psvr_rmspe()`](https://pbenavidesh.github.io/psvr/reference/psvr_rmspe.md)
and the parsnip engine fit wrappers both return with `sym_type = "even"`
/ `"odd"`. The kernel must satisfy Assumption 3 of the paper (kernel
symmetry); see
[`make_kernel()`](https://pbenavidesh.github.io/psvr/reference/make_kernel.md).

## Usage

``` r
.fit_rmspe_sym(X, y, kernel, gamma, a = 1, precondition = "auto")
```

## Arguments

- X, y, kernel, gamma, precondition:

  See
  [`psvr_rmspe()`](https://pbenavidesh.github.io/psvr/reference/psvr_rmspe.md)
  for the full semantics of each argument (including the Remark-17
  preconditioner).

- a:

  Symmetry type: `1` (even) or `-1` (odd). This is the internal integer;
  the public argument is `sym_type`, with `"even"` mapping to `a = 1`
  and `"odd"` to `a = -1`. Neither
  [`psvr_rmspe()`](https://pbenavidesh.github.io/psvr/reference/psvr_rmspe.md)
  nor the parsnip specifications expose `a` directly.

## Value

A list of class `"psvr_rmspe_sym"`.
