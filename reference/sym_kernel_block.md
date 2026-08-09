# Compute a symmetric kernel block for prediction

Block form of
[`sym_kernel_vector()`](https://pbenavidesh.github.io/psvr/reference/sym_kernel_vector.md):
entry `[k, i]` equals `½ * Ks(X[k, ], Xnew[i, ])` where
`Ks(xi, xj) = K(xi, xj) + a * K(xi, -xj)`. Used by the
[`predict()`](https://rdrr.io/r/stats/predict.html) methods of the
symmetric models (Models 2 and 4).

## Usage

``` r
sym_kernel_block(K, X, Xnew, a)
```

## Arguments

- K:

  A kernel function from
  [`make_kernel()`](https://pbenavidesh.github.io/psvr/reference/make_kernel.md).

- X:

  Numeric training matrix (N × p).

- Xnew:

  Numeric matrix of new points (M × p).

- a:

  Symmetry parameter: `1` (even) or `-1` (odd).

## Value

Numeric N × M matrix.

## Details

Built from two
[`kernel_matrix()`](https://pbenavidesh.github.io/psvr/reference/kernel_matrix.md)
calls rather than a nested R loop, so built-in kernels reach the Rcpp
implementations; user-defined closures still fall through to
[`.legacy_kernel_matrix()`](https://pbenavidesh.github.io/psvr/reference/dot-legacy_kernel_matrix.md).
Element-wise this is the same `0.5 * (K + a * K_neg)` arithmetic in the
same order as the per-row loop it replaces, so predictions are
bit-identical.
