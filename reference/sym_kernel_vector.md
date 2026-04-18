# Compute a symmetric kernel vector for prediction

For a new point `x`, returns the N-vector with entry `k` equal to
`½ * Ks(X[k, ], x)` where `Ks(xi, xj) = K(xi, xj) + a * K(xi, -xj)`.
Used by
[`predict.psvr_mape_sym()`](https://pbenavidesh.github.io/psvr/reference/predict.psvr_mape_sym.md)
and
[`predict.psvr_rmspe_sym()`](https://pbenavidesh.github.io/psvr/reference/predict.psvr_rmspe_sym.md).

## Usage

``` r
sym_kernel_vector(K, X, x, a)
```

## Arguments

- K:

  A kernel function from
  [`make_kernel()`](https://pbenavidesh.github.io/psvr/reference/make_kernel.md).

- X:

  Numeric training matrix (N × p).

- x:

  Numeric vector (length p), the new point to predict.

- a:

  Symmetry parameter: `1` (even) or `-1` (odd).

## Value

Numeric vector of length N.
