# Compute the symmetrized kernel matrix Ωs = ½(Ω + a·Ω\*)

Used by the symmetric LS-SVR model (Model 4). Entry `[k, l]` of `Ω*` is
`K(xk, -xl)`, so negation is applied to the columns of X (i.e., to
`X2`).

## Usage

``` r
sym_kernel_matrix(K, X, a)
```

## Arguments

- K:

  A kernel function from
  [`make_kernel()`](https://pbenavidesh.github.io/psvr/reference/make_kernel.md).

- X:

  Numeric training matrix (N × p).

- a:

  Symmetry parameter: `1` (even) or `-1` (odd).

## Value

Numeric N × N matrix `Ωs = ½(Ω + a·Ω*)`.
