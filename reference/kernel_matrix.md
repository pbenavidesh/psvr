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
