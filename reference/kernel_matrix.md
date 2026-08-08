# Compute a kernel matrix between two sets of points

Entry `[i, j]` equals `K(X1[i, ], X2[j, ])`. Used internally by all four
model fitting and prediction functions.

## Usage

``` r
kernel_matrix(K, X1, X2 = X1)
```

## Arguments

- K:

  A kernel function from
  [`make_kernel()`](https://pbenavidesh.github.io/psvr/reference/make_kernel.md).

- X1:

  Numeric matrix with one observation per row (n1 × p).

- X2:

  Numeric matrix with one observation per row (n2 × p). Defaults to
  `X1`, giving the square training kernel matrix Ω.

## Value

Numeric matrix of size n1 × n2.

## Details

If `K` was produced by
[`make_kernel()`](https://pbenavidesh.github.io/psvr/reference/make_kernel.md),
dispatch reads the `kernel_info` attribute and calls the Rcpp
implementation for the three built-in types (`"rbf"`, `"linear"`,
`"polynomial"`). For user-defined closures (no `kernel_info` attribute)
the dispatch falls through to
[`.legacy_kernel_matrix()`](https://pbenavidesh.github.io/psvr/reference/dot-legacy_kernel_matrix.md),
the original pure-R nested loop. Predictions are bit-identical to the
R-only path on Windows/Rtools45; see `src/kernel_*.cpp` for the
operation-order rationale.
