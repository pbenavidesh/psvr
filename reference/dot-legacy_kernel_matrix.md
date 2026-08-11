# Pure-R nested-loop kernel matrix (fallback path)

Original
[`kernel_matrix()`](https://pbenavidesh.github.io/psvr/reference/kernel_matrix.md)
body, retained as a fallback for kernel closures that do not carry a
`kernel_info` attribute (i.e., not built via
[`make_kernel()`](https://pbenavidesh.github.io/psvr/reference/make_kernel.md)).
Also used by tests to verify Rcpp-vs-R parity.

## Usage

``` r
.legacy_kernel_matrix(K, X1, X2)
```

## Arguments

- K:

  A kernel function from
  [`make_kernel()`](https://pbenavidesh.github.io/psvr/reference/make_kernel.md).

- X1:

  Numeric matrix with one observation per row (n1 × p).

- X2:

  Numeric matrix with one observation per row (n2 × p). Defaults to
  `X1`, giving the square training kernel matrix \\\Omega\\.

## Value

Numeric matrix of size n1 × n2.
