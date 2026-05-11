# F7-C-full — Portable C++ architecture

**Status:** Archived implementation notes — current state lives in `CLAUDE.md`.

## Portable C++ architecture (post-F7-C-full)

### Motivation

The R-level F7 implementation had a 2× per-iter wall overhead from
R-language scaffolding (vector subsetting, `which.max`, `setdiff`,
`ifelse`). Iter reduction (38–48%) did not translate to wall
positivity. A C++ port was undertaken to (a) eliminate the R
scaffolding, restoring wall positivity for T7, and (b) demonstrate
the paper's portability claim by architecting the SMO core such that
a future Python binding (pybind11) wraps the same C++ without R-isms.

### Architecture overview

```
src/
  core_smo_types.h          types-only header (FitOptions, FitResult, Vec)
  core_smo_solve.h/.cpp     SMO outer loop, WSS1/WSS3, bias recovery
  core_block_k4.h/.cpp      Theorem 7 selection + descent test + 2-D update
  core_kernel.h
  core_kernel_rbf.cpp       extracted from F6 src/kernel_rbf.cpp
  core_kernel_linear.cpp    extracted from F6 src/kernel_linear.cpp
  core_kernel_poly.cpp      extracted from F6 src/kernel_poly.cpp
  binding_smo.cpp           [[Rcpp::export]] psvr_smo_fit_rcpp
  binding_kernel.cpp        [[Rcpp::export]] kernel_*_cpp
  RcppExports.cpp           auto-generated
  Makevars / Makevars.win   PKG_LIBS = $(BLAS_LIBS) $(FLIBS)
```

The `core_*.cpp` files use only `std::vector`, `std::ptrdiff_t`,
`long double` accumulators, `std::pow`. No `Rcpp::*`, no `Rcout`, no
`R_xlen_t`. The only R-API dependency in the core is `F77_CALL(dgemv)`
(via `<R_ext/BLAS.h>` in the R-build path; via the `dgemv_` extern in
the standalone path — see "Conditional compilation" below).

### Memory layout

`Omega` is column-major (matching R's matrix storage). For a sample
index `p ∈ [0, N)`, the column `K_p` is the contiguous slice
`Omega[p*N .. p*N + N - 1]`. `Omega(i, j)` is at `Omega[j*N + i]`.
Zero-copy from R: `REAL(NumericMatrix)` returns the underlying
column-major `double*`.

### Memory ownership

All `double*` and `Index*` arguments to core functions are
**caller-owned**. The core never allocates output buffers; it writes
into `std::vector`s declared in `FitResult` (which the binding then
copies into Rcpp::NumericVectors for return to R).

### BLAS access

The matvec `Kbeta = Omega · β` runs three times per fit (warm-start
init, unshrink-rebuild inside the loop, post-loop refresh).
`core_smo_solve.cpp` calls `F77_CALL(dgemv)` with `FCONE` so R's BLAS
is used directly — this matches `R/.smo_solve_r()`'s `Omega %*% v`
bit-identically (the operation goes through the same BLAS impl).

### Conditional compilation pattern (paper TODO #11)

```cpp
#ifdef PSVR_STANDALONE_BUILD
extern "C" void dgemv_(...);
#define F77_CALL(x) x ## _
#define FCONE
#else
#include <R_ext/BLAS.h>
#include <R_ext/RS.h>
#endif
```

The R-build path uses R's BLAS via `<R_ext/BLAS.h>`. The standalone
path (`-DPSVR_STANDALONE_BUILD`) is used by `dev/check_core.cpp` to
verify the core compiles without R headers. Future pybind11 binding
will add a third arm (`#elif defined(PSVR_PYTHON_BUILD)`) backed by
numpy's BLAS — same pattern, demonstrating the portability claim.

### Bit-identicality discipline

Bit-equality with the R reference (`engine = "r"`) is the strict
gate. Discipline in the C++ port:

- **Loop direction** matches R's evaluation order (e.g.,
  `down_alpha` is scanned before `down_astar` in WSS3 candidate
  search; `R` does `c(down_alpha, down_astar)`).
- **`which.max` tie-break**: R returns the FIRST index of the
  maximum. C++ uses strict `>` (not `>=`) in the tracked-best loop
  to preserve this semantic. Comments at scan sites document the
  invariant.
- **Long-double accumulators** where R uses LDOUBLE (e.g.,
  `mean(y)`, kernel matrix inner reductions).
- **Separate subtractions, not fused arithmetic** for the joint
  τ update. R writes `tau - δ_1 * diff_1 - δ_2 * diff_2` (two left-
  associative subtractions). Fused `tau -= (δ_1 * diff_1 + δ_2 *
  diff_2)` would round differently (1 ulp). The C++ port uses two
  separate `tau -= ...` statements. (This was caught during Phase 2
  STOP 2: an 8.88e-16 drift triggered the escalation policy and was
  fixed before merging.)

`tests/testthat/test-engine-equivalence.R` is the 16-config canary
that locks the invariants. The `.diagnose_engine_diff()` helper
prints max diff + side-by-side values at full precision on failure
to support the FP-tier escalation policy.

### Build system notes

`src/Makevars` and `src/Makevars.win` specify:

```
PKG_LIBS = $(BLAS_LIBS) $(FLIBS)
```

This is required because `core_smo_solve.cpp` calls
`F77_CALL(dgemv)` for the matvec. Without `$(BLAS_LIBS)` the package
fails to link with `undefined reference to dgemv_`.

**Do not remove this line.** `$(FLIBS)` handles Fortran runtime
linking on platforms where BLAS implementations need it.

**Do not add `$(LAPACK_LIBS)`** unless a future revision actually
uses LAPACK routines. R CMD check warns about
"`$(LAPACK_LIBS)` without following `$(BLAS_LIBS)`" when LAPACK is
included unnecessarily. Per *Writing R Extensions* §1.2.3, the
correct order when both are needed is:

```
PKG_LIBS = $(LAPACK_LIBS) $(BLAS_LIBS) $(FLIBS)
```

(LAPACK first, since the linker resolves dependencies right-to-left
and LAPACK depends on BLAS.) The current code uses only BLAS dgemv;
there is no LAPACK dependency.
