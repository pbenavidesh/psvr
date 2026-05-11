# F6 — Rcpp-accelerated kernel construction

**Status:** Archived implementation notes — current state lives in `CLAUDE.md`.

## Rcpp-accelerated kernel construction (post-F6)

F6 attacks the F5 hot path: `kernel_matrix()` consuming ~89% of wall-
clock and producing a ~28× memory overhead (R-level intermediates from
the vectorized nested loop) for an 8 MB output at N=1000. The paper's
Theorem 6 (LIBSVM column-on-demand cache) targets a different
architecture (column streaming), not psvr's full-matrix design, so we
**skip T6** and instead replace the matrix construction itself with a
C++ implementation, plus a cross-fold reuse path in `psvr_cv()`.

### Architecture

- **Three Rcpp kernels** in `src/kernel_rbf.cpp`, `kernel_linear.cpp`,
  `kernel_poly.cpp`. The `[[Rcpp::export]]` symbols
  `kernel_rbf_cpp(X1, X2, sigma)`, `kernel_linear_cpp(X1, X2)`, and
  `kernel_poly_cpp(X1, X2, coef0, degree)` are internal — callers reach
  them only through `kernel_matrix()`.
- **`kernel_matrix(K, X1, X2 = X1)`** in `R/kernel.R` reads
  `attr(K, "kernel_info")` (set by `make_kernel()` and exposing
  `type, sigma, degree, coef0`) and dispatches via `switch(info$type, …)`.
  User-defined closures (no `kernel_info`) fall through to
  `.legacy_kernel_matrix()`, the original pure-R nested loop. The
  single swap point covers all four fitters + `sym_kernel_matrix()` +
  the predict path (`utils-predict.R`).
- **`precomputed_Omega` / `precomputed_Omega_s`** internal parameters on
  `psvr()`, `.fit_mape()`, `.fit_mape_sym()` allow callers to bypass
  the per-call kernel build. Diagonal jitter (`1e-6` for non-sym,
  `0.5e-6` for sym) and the adaptive spectral shift still run on the
  (subset of the) precomputed matrix. Precomputed matrices must be
  **un-jittered**.
- **`psvr_cv()`** (rset path) computes `Omega_full` or `Omega_s_full`
  once over the union of `in_id` across all splits (`data_full <-
  splits$splits[[1]]$data`), then slices
  `Omega_full[row_ids_i, row_ids_i]` per fold. The list-of-tuples path
  falls back to per-fold construction (still benefits from the per-call
  Rcpp dispatch). LS-SVR (`loss = "rmspe"`) is already gated out of
  `psvr_cv()`.

### Bit-identicality with the legacy R path

Three deliberate choices in C++ preserve bit-equality on Windows /
Rtools45 with the legacy R nested loop:

1. **Long-double accumulator** in all three kernels' inner reduction
   (`sum-of-squared-diffs` for RBF, dot product for linear and
   polynomial). R's `sum()` in `src/main/summary.c` uses
   `LDOUBLE s = 0.0; s += x[i];` (80-bit extended precision on x87 via
   mingw-w64). A naive `double` accumulator drifts at ~5e-17 for RBF
   and ~2e-15 for linear; the long-double version is bit-equal.
2. **`powl` for polynomial** when `degree >= 3`:
   `(double) std::pow((long double) base, (long double) degree)`
   mirrors R's `R_POW` under `HAVE_LONG_DOUBLE` (arithmetic.c calls
   `(double) powl((LDOUBLE) x, (LDOUBLE) y)`). `degree == 2` falls
   through to `base * base` (R_POW's special case). Plain
   `std::pow(double, double)` drifts at ~9e-13.
3. **No self-kernel symmetry shortcut** — the legacy R loop computes
   the full upper + lower triangles, so we do too. Bit-symmetry of the
   output is preserved naturally by IEEE 754 arithmetic.

Verified by the snapshot gate: both `_snaps/bit-identical.md` and
`_snaps/psvr-direct.md` MD5s are unchanged from F5 baseline
(`D3FBB28C…` and `C3F3467924…` respectively). `test-rcpp-kernels.R`
adds 14 elementwise-`identical()` tests at N ∈ {10, 100, 200, 500}
across all three kernels and polynomial degrees {1, 2, 3, 4}.

### `psvr_cv()` cross-fold reuse

`row_ids_i <- split_i$in_id` (already used for warm-start alignment)
also serves as the kernel-matrix slice index. `match()`-based mapping
is unchanged. The F5 warm-start handoff (`new_mask`, `alpha_init`,
`alpha_star_init`) is preserved exactly. `test-psvr-cv-reuse.R`
verifies bit-identical fits between the precomputed-Omega path and a
manual per-fold loop without precompute.

### Performance characteristics (Windows / R 4.5.3 / Rtools45)

The long-double accumulator costs ~5× vs a hypothetical double-only
Rcpp implementation (x87 FPU on this platform). The trade-off keeps
the snapshot gate intact. Bench numbers (`dev/bench-F6.R`, RBF,
N × 5 random Gaussian X):

- `kernel_matrix(N=1000)`:  pre-F6 ~2.4 s  → post-F6 ~0.2 s  (~12× wall)
- `kernel_matrix(N=3000)`:  pre-F6 ~21 s   → post-F6 ~1.8 s  (~12× wall)
- `kernel_matrix(N=10000)`: pre-F6 NOT MEASURED (would be ~5 min in
  pure R) → post-F6 ~21 s.

Cross-fold reuse in `psvr_cv()` adds a smaller benefit (~1.05× on the
10-fold N=1000 fixture); the SMO solve dominates total wall time, so
the precomputation savings are absolute (~1.5 s per 10-fold call) but
small relative to per-fold SMO. The user-visible win is the per-call
Rcpp dispatch.

### Why paper Theorem 6 is skipped

Theorem 6's LIBSVM-style column-on-demand cache is designed for an
SMO that requests *individual columns* of `Ω` per iteration and reuses
them across iterations under an LRU policy. psvr's
`.make_kernel_accessor()` (post-F2) is a thin closure over an already-
materialised matrix; the SMO loop reads columns from RAM, never
synthesizing on demand. Implementing T6 would require either (a)
abandoning the full-matrix design (~architectural rewrite of F2) or
(b) building T6 as a *replacement* for `.make_kernel_accessor()` whose
backing store is column synthesis. (a) is out of F6 scope; (b) saves
no memory in the typical `kernel_matrix(K, X)` pre-construction path
since the full matrix is built up front anyway. F6 instead targets
the matrix construction itself (`O(N²·p)` work, the actual hot path)
via Rcpp, and the result holds the matrix in RAM. See paper TODO #8
for the recommended rewrite in the paper text.
