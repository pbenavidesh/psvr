# Build a kernel-matrix accessor (internal)

Wraps a fully-materialised kernel matrix in a list of accessor closures.
The SMO solver and other iterative consumers read kernel values through
this interface so that future phases can replace the underlying
representation (e.g., the F6 LRU cache backed by Rcpp) without touching
the consumers.

## Usage

``` r
.make_kernel_accessor(Omega)
```

## Arguments

- Omega:

  Kernel matrix (`N x N`), already with any required jitter and
  symmetrisation applied by the caller.

## Value

A list with components:

- `get_column(p)`:

  Returns column `p` of `Omega` (length-`N` numeric vector).

- `get_diag()`:

  Returns `diag(Omega)` (length-`N` numeric vector). Cached at
  construction.

- `get_entry(p, q)`:

  Returns the scalar `Omega[p, q]`.

- `get_matvec(v)`:

  Returns the matrix-vector product `as.numeric(Omega %*% v)`
  (length-`N`). Used for one-shot gradient refreshes; preserves BLAS
  efficiency.

- `n`:

  Integer, equal to `nrow(Omega)`.

## Details

This F2 implementation is a thin wrapper over the materialised matrix:
every closure delegates directly to base-R indexing or BLAS. The
diagonal is computed once at construction
([`diag()`](https://rdrr.io/r/base/diag.html) is called a single time)
and reused on every `get_diag()` call.

**Symmetry assumption.** The wrapped matrix is assumed symmetric. Both
\\\Omega\\ (a kernel matrix for symmetric `K`) and \\\Omega_s =
\tfrac{1}{2}(\Omega + a\\\Omega^\*)\\ (used by the symmetric models when
`K` satisfies Assumption 3) are symmetric throughout this package, so
callers may use `get_column(p)[k]` in place of a row read `Omega[p, k]`.
The SMO solver relies on this in its WSS3 step.
