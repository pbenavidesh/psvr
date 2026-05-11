# psvr — Percentage-error Support Vector Regression

## Package purpose

`psvr` is an R package implementing four SVR models derived from a unified mathematical
framework for percentage-error loss functions. It targets forecasting contexts where
targets `yk > 0` are strictly positive and relative accuracy matters more than absolute
accuracy.

Source paper: *A Unified Family of Percentage-Error Support Vector Regression Models with
Symmetric Kernel Extensions* (MDPI Mathematics, 2026). Full derivations are in `paper.tex`
at the repository root.

---

## Mathematical notation

| Symbol | Meaning |
|--------|---------|
| N | number of training samples |
| `xk ∈ Rᵖ` | input vector for sample k |
| `yk > 0` | strictly positive target (Assumption 2 in paper) |
| `K(xi, xj)` | kernel function |
| `C > 0` | regularization parameter (ε-SVR models) |
| `ε ≥ 0` | insensitivity tube width (ε-SVR models) |
| `Γ > 0` | regularization parameter (LS-SVR models) |
| `a ∈ {-1, 1}` | symmetry type: 1 = even, -1 = odd |
| `βk = αk - αk*` | dual variable differences (ε-SVR) |
| `αk ∈ R` | dual / Lagrange multipliers (LS-SVR) |
| `b` | bias term |
| `Ω ∈ R^{N×N}` | kernel matrix, `Ωkl = K(xk, xl)` |
| `Ωs = ½(Ω + aΩ*)` | symmetrized kernel matrix |
| `Ω*kl = K(xk, -xl)` | cross-negation kernel matrix |
| `Ks(xi,xj) = K(xi,xj) + a·K(xi,-xj)` | symmetric kernel |
| `YΓ = diag(y1²/Γ, …, yN²/Γ)` | target-weighted regularization diagonal |

---

## The four models

### Model 1 — ε-SVR with MAPE (Theorem 1 / Appendix A.1)

**Primal:**
```
min  ½‖ω‖² + C Σk (ξk + ξk*)
s.t. (100/yk)(yk - ωᵀφ(xk) - b) ≤ ε + ξk
     (100/yk)(ωᵀφ(xk) + b - yk) ≤ ε + ξk*
     ξk, ξk* ≥ 0
```

**Dual QP (max form):**
```
max  -½ Σk,l βk βl K(xk,xl) + Σk βk yk - (ε/100) Σk |βk| yk
s.t. Σk βk = 0
     |βk| ≤ 100C / yk    (sample-dependent box; yk small → tight bound)
```
where `βk = αk - αk*`, `|βk| = αk + αk*`.

**KKT:** `ω = Σk βk φ(xk)`, `Σk βk = 0`, `0 ≤ αk, αk* ≤ 100C/yk`.

**Prediction:** `f(x) = Σk βk K(xk, x) + b`

**Implementation:** QP via `osqp`. Key files: `R/mape_svr.R`.

---

### Model 2 — Symmetric ε-SVR with MAPE (Theorem 2 / Appendix A.2)

Same primal as Model 1, augmented with symmetry constraint:
`ωᵀφ(xk) = a · ωᵀφ(-xk)`

**Dual QP (max form):**
```
max  -¼ Σk,l βk βl Ks(xk,xl) + Σk βk yk - (ε/100) Σk |βk| yk
s.t. Σk βk = 0
     |βk| ≤ 100C / yk    (unchanged from Model 1)
```
where `Ks(xi,xj) = K(xi,xj) + a·K(xi,-xj)`.

The factor `¼` (vs `½` in Model 1) arises from the symmetrization in the representer
theorem (Appendix A.2): `ωᵀφ(xk) = ½ Σl βl Ks(xl, xk)`, so the quadratic term picks
up an extra `½`.

**Prediction:** `f(x) = ½ Σk βk Ks(xk, x) + b`

**Implementation:** QP via `osqp`. Build `Ks = Ω + a·Ω*` inline (no `½`), then pass
`P = ¼·Ks` as the QP quadratic matrix — the `¼` factor is absorbed into `P` directly.
Do **not** use `sym_kernel_matrix()` here; that helper returns `½(Ω + a·Ω*)` = `Ωs`,
which is only correct for Model 4. Key files: `R/mape_sym_svr.R`.

---

### Model 3 — LS-SVR with RMSPE (Theorem 3 / Appendix A.3)

**Primal:**
```
min  ½‖ω‖² + (Γ/2) Σk ek²/yk²
s.t. yk = ωᵀφ(xk) + b + ek
```

**KKT stationarity:**
- `ω = Σk αk φ(xk)`
- `Σk αk = 0`
- `ek = (yk²/Γ) αk`

**Linear system (solve directly):**
```
[ 0    1ᵀ      ] [b]   [0]
[ 1    Ω + YΓ  ] [α] = [y]
```
where `YΓ = diag(y1²/Γ, …, yN²/Γ)` — added to the diagonal of Ω only.

**Prediction:** `f(x) = Σk αk K(xk, x) + b`

**Implementation:** `base::solve()` on (N+1)×(N+1) augmented matrix.
Key files: `R/rmspe_lssvr.R`.

---

### Model 4 — Symmetric LS-SVR with RMSPE (Theorem 4 / Appendix A.4)

Same primal as Model 3, augmented with symmetry constraint.

**Linear system:**
```
[ 0    1ᵀ       ] [b]   [0]
[ 1    Ωs + YΓ  ] [α] = [y]
```
where `Ωs = ½(Ω + aΩ*)`, `Ω*kl = K(xk, -xl)`.

**Symmetric representer:** `ωᵀφ(xk) = Σl (Ωs)kl αl`

**Prediction:** `f(x) = ½ Σk αk Ks(xk, x) + b`
equivalently: `f(x) = Σk αk [K(xk,x) + a·K(xk,-x)] / 2 + b`

**Implementation:** `base::solve()` with `Ωs` replacing `Ω`.
Key files: `R/rmspe_sym_lssvr.R`.

---

## Kernel symmetry (Assumption 3 in paper)

Required for Models 2 and 4:
- `K(-xi, xj) = K(xi, -xj)`
- `K(-xi, -xj) = K(xi, xj)`

**RBF kernel** `K(xi, xj) = exp(-‖xi - xj‖²/2σ²)` satisfies both conditions:
- `K(-xi, -xj) = exp(-‖-xi-(-xj)‖²/2σ²) = exp(-‖xi-xj‖²/2σ²) = K(xi,xj)` ✓
- `K(-xi, xj) = exp(-‖-xi-xj‖²/2σ²) = exp(-‖xi+xj‖²/2σ²) = K(xi,-xj)` ✓

**Polynomial kernels of even degree** also satisfy Assumption 3.

---

## Implementation decisions

- **QP solver:** `osqp` package (Models 1 & 2). Supports sparse matrices, no commercial license.
- **Linear solver:** `base::solve()` (Models 3 & 4). Dense (N+1)×(N+1) system, no extra dep.
- **Target validation:** all `fit()` functions call `stopifnot(all(y > 0))`.
- **Kernel interface:** `make_kernel(type, ...)` returns a closure `function(xi, xj)` from
  type ∈ `{"rbf", "linear", "polynomial"}`. Shared across all four models. As of F6 the
  closure carries `attr(K, "kernel_info") = list(type, sigma, degree, coef0)` and
  `kernel_matrix()` dispatches through this to the Rcpp implementation
  (`src/kernel_*.cpp`); user-defined closures without the attribute fall through to the
  R-only `.legacy_kernel_matrix()`.
- **YΓ storage:** stored as a vector `y^2 / gamma`; added to diagonal via
  `diag(Omega) <- diag(Omega) + y_gamma` — never materializes a full N×N diagonal matrix.
- **S3 classes:**
  - `"psvr_fit"` — single class returned by [psvr()] (post-F1 unified API).
  - Legacy classes `"psvr_mape"`, `"psvr_mape_sym"`, `"psvr_rmspe"`,
    `"psvr_rmspe_sym"` are still returned by the deprecated wrappers in
    `R/deprecated.R`; their `predict`/`print`/`coef` methods are still
    registered.
- **S3 methods:** `predict()`, `print()`, `coef()`, `summary()` on
  `psvr_fit`; `predict()` / `print()` / `coef()` on each legacy class.
- **Documentation:** roxygen2 with `@param`, `@return`, `@examples`.
- **Style:** tidyverse / base R style: `snake_case`, `<-` assignment.
- **License:** MIT.

---

## Implementation order

1. [x] `CLAUDE.md` — this file
2. [x] Package scaffolding: `DESCRIPTION`, `NAMESPACE`, directory structure
3. [x] `R/kernel.R` — `make_kernel()` helper
4. [x] `R/rmspe_lssvr.R` — Model 3: now `.fit_rmspe()` + `predict.psvr_rmspe()`
5. [x] `R/rmspe_sym_lssvr.R` — Model 4: now `.fit_rmspe_sym()` + `predict.psvr_rmspe_sym()`
6. [x] `R/mape_svr.R` — Model 1: now `.fit_mape()` + `predict.psvr_mape()`
7. [x] `R/mape_sym_svr.R` — Model 2: now `.fit_mape_sym()` + `predict.psvr_mape_sym()`
8. [x] `tests/` — testthat unit tests for all four models (84 tests, 0 failures)
9. [x] **F1** — Unified `psvr()` entry point + `psvr_fit` class + DRY consolidation +
       symmetric-kernel standardization (`R/psvr-main.R`, `R/psvr-methods.R`,
       `R/utils-*.R`, `R/deprecated.R`).
10. [x] **F2** — Kernel accessor interface (`R/kernel-accessor.R`); SMO
        loop reads kernel data via `K_acc` closures. Foundation for F3/F6.
11. [x] **F3** — Algorithm 2: adaptive spectral shift
        (`R/kernel-spectral.R`). `.fit_mape_sym()` now invokes
        `.adaptive_spectral_shift()` between the diagonal-jitter step and
        the SMO/osqp solve; spectral diagnostics surface under
        `solver_meta$spectral` on `psvr_fit` objects.
12. [x] **F4** — Theorems 3 + 8 from arXiv:2605.01446 v3 (asymmetric freeze
        per-sample thresholds + per-pair tolerance scaling). Modifies
        `R/smo_solve.R` only. Bit-identicalidad with F3 NOT preserved on
        heterogeneous-target data (intentional); homogeneous regime
        collapses to F3 baseline. Iter reduction ~25% on rho_y >> 50
        datasets.
13. [x] **F5** — Theorem 5 (warm-start API with new-samples-only Algorithm
        1 Step 2 deviation) + `psvr_cv()` CV helper + `fit$alpha` →
        `fit$beta` rename. Theorem 4 evaluation: Fan-Chen-Lin WSS3
        unconstrained (libsvm-style) is empirically sufficient for
        MAPE-SVR; both the original heuristic (`alpha_wss` multiplier)
        and the Glasmachers-Igel max-gain WSS were evaluated and found
        to provide no benefit. See paper TODOs #5, #6, #7 below.
14. [x] **F6** — Rcpp-accelerated `kernel_matrix()` for RBF / linear /
        polynomial (`src/kernel_*.cpp`, `R/kernel.R` dispatch) +
        cross-fold kernel reuse in `psvr_cv()` via internal
        `precomputed_Omega` / `precomputed_Omega_s` channel
        (`R/mape_svr.R`, `R/mape_sym_svr.R`, `R/psvr-main.R`,
        `R/psvr_cv.R`). Bit-identical to F5 baseline (snapshot MD5s
        unchanged). Paper Theorem 6 (LIBSVM column-on-demand cache) was
        skipped — architectural mismatch with psvr's full-matrix design.
        See paper TODO #8 below.
15. [ ] **F7–F8** — Theorem 7 from arXiv:2605.01446 v3 + paper-side
        errata fixes (#1–#8).

---

## Architecture (post-F1)

### Public API

- **`psvr(X, y, loss, sym, kernel, ...)`** is the single entry point.
  Selects among the four model families via `loss = "mape" | "rmspe"` and
  `sym = NULL | +1L | -1L`. Returns a `psvr_fit` object. See `R/psvr-main.R`.
- **`psvr_fit` class** with methods: `predict()`, `print()`, `coef()`, `summary()`.
  Field schema:
  ```
  list(loss, sym, kernel, alpha, b, support_data, support_targets,
       n_train, n_sv, p_train, hyperparameters, solver_meta)
  ```
  `alpha` carries `α − α* = β` for ε-SVR and the LS-SVR `α` for LS-SVR.
  `support_data` is `X_sv` (post-pruning) for ε-SVR or full `X` for LS-SVR.
  `support_targets` is non-NULL only for ε-SVR.
- **12 parsnip specs** (`psvr_mape_rbf()`, `psvr_rmspe_sym_poly()`, etc.) —
  unchanged. Frozen as public API.
- **dials helpers** (`cost_psvr`, `margin_percentage`, `rbf_sigma_psvr`,
  `sym_type_param`, etc.) — unchanged.

### Deprecated public API (to be removed in v0.2.0+)

- `mape_svr()`, `mape_sym_svr()`, `rmspe_lssvr()`, `rmspe_sym_lssvr()` —
  thin wrappers in `R/deprecated.R` that emit `.Deprecated("psvr")` and
  delegate directly to the corresponding `.fit_*` internal. They preserve
  the OLD object shapes (`psvr_mape`, `psvr_mape_sym`, `psvr_rmspe`,
  `psvr_rmspe_sym`) and the OLD predict/print/coef methods continue to
  dispatch correctly on those classes.

### Internal helpers

- **Fitters**: `.fit_mape()`, `.fit_mape_sym()`, `.fit_rmspe()`,
  `.fit_rmspe_sym()` — bodies of the four legacy public fitters, now
  internal. Return the OLD shape with the OLD class.
- **`R/utils-validation.R`**: `.validate_y_positive()`, `.warn_large_n()`,
  `.validate_psvr_inputs()` (the latter uses the `missing()`-flag pattern
  for cross-loss param warnings, since `match.arg` defaults are otherwise
  indistinguishable from user input).
- **`R/utils-precondition.R`**: `.resolve_precondition()`.
- **`R/utils-print.R`**: `.kernel_desc()`.
- **`R/utils-predict.R`**: `.psvr_predict_dispatch()` — per-row predict
  loop for `psvr_fit` objects, branching on `is.null(sym)`.
- **12 parsnip fit wrappers** (`psvr_mape_rbf_fit()` etc.): tagged
  `@keywords internal` (hidden from pkgdown reference) but still exported.
  Parsnip's `set_fit()` resolves `c(pkg, fun)` via `pkg::fun` (i.e.
  `getExportedValue`), which only sees exported objects, so the leading-
  dot-internal rename does not work. The visibility demotion is therefore
  cosmetic: callers can still reach them as `psvr::psvr_mape_rbf_fit(...)`,
  but they are not advertised as user API.

### Unified Ωs (with ½) symmetric-kernel convention

All symmetric-kernel code paths now build `Ωs = ½(Ω + a·Ω*)` via
`sym_kernel_matrix()` and pass `Ωs` directly to the solver. Both Models 2
and 4 use this convention; the SMO/osqp/QP code does NOT need to apply a
`0.5 *` factor to the input matrix any more.

Bit-identicality with the pre-F1 path is preserved at tolerance `1e-10`
on a 16-test golden snapshot suite (`tests/testthat/test-bit-identical.R`)
and a 12-test direct-`psvr()` suite (`tests/testthat/test-psvr-direct.R`).
Model 2's diagonal jitter is set to `0.5e-6` (not `1e-6`) — a deliberate
carve-out — to preserve bit-identicality with the pre-F1 path where
`diag(Ks) += 1e-6` was followed by `0.5 * Ks`. See `R/mape_sym_svr.R` for
the inline rationale.

### Routing

`psvr()` dispatches via:
```r
fit <- switch(paste(loss, ifelse(is.null(sym), "std", "sym"), sep = "_"),
  mape_std  = .fit_mape(X, y, kernel, C, eps, solver, tol),
  mape_sym  = .fit_mape_sym(X, y, kernel, C, eps, a, solver, tol),
  rmspe_std = .fit_rmspe(X, y, kernel, gamma, precondition),
  rmspe_sym = .fit_rmspe_sym(X, y, kernel, gamma, a, precondition))
```
followed by an OLD→NEW shape rewrap. The deprecation wrappers in
`R/deprecated.R` skip this rewrap and return the `.fit_*` output as-is.
Net effect: shape translation lives in exactly one place.

---

## Kernel Accessor (post-F2)

The SMO inner loop reads kernel values through an **accessor closure**
built by `.make_kernel_accessor(Omega)` in `R/kernel-accessor.R`. The
accessor is a list with five components:

```
get_column(p)  -> Omega[, p]               (length-N numeric)
get_diag()     -> diag(Omega)              (cached at construction)
get_entry(p,q) -> Omega[p, q]              (scalar)
get_matvec(v)  -> as.numeric(Omega %*% v)  (length-N; preserves BLAS)
n              -> nrow(Omega)              (integer)
```

**Why.** Decoupling the SMO solver from the matrix representation is
the foundation for the F3–F7 efficiency theorems:
- F3 — Algorithm 2 (adaptive spectral shift): the accessor can wrap a
  spectrally-shifted kernel without touching the solver.
- F6 — adaptive LRU cache via Rcpp (Theorem 6): the closure-based
  wrapper is replaced with a cache-backed implementation; the SMO loop
  and fitters do not change.
- F7 — block-`k=4` working set (Theorem 7): the per-iteration column
  fetch generalises to a block fetch through the same interface.

**Symmetry invariant.** The wrapped matrix is assumed symmetric. Both
`Ω` (a kernel matrix for symmetric `K`) and `Ωs = ½(Ω + a·Ω*)` (the
unified symmetric-kernel matrix from `sym_kernel_matrix()`, valid when
`K` satisfies Assumption 3) are symmetric throughout this package, so
the SMO loop substitutes `K_acc$get_column(p)[k]` for the row read
`Omega[p, k]` (used in WSS3).

**Current implementation.** F2 ships a thin closure wrapper over the
fully materialised matrix:
- Two column fetches per inner SMO iteration: `K_p` once after WSS1
  picks `p` (reused for WSS3, the 1-D step size, and the gradient
  update) and `K_q` once after WSS3 picks `q` (used for the gradient
  update only).
- Diagonal cached at construction (`diag()` called once).
- The two full matvecs (the shrink-rebuild path inside the loop and the
  post-loop tau refresh) go through `get_matvec()`, which delegates to
  `as.numeric(Omega %*% v)` so BLAS is preserved.
- No per-cell `get_entry()` access inside the inner loop.

**Scope.** The accessor is used by Models 1 and 2 only (the SMO path).
LS-SVR fitters (`.fit_rmspe`, `.fit_rmspe_sym`) solve a single linear
system via `base::solve()` and do **not** use accessors — caching is not
applicable to a one-shot factorisation.

**Future swap point.** F6 will replace `.make_kernel_accessor()` with a
cache-backed implementation (likely Rcpp). The five-method contract is
the only API surface the SMO solver depends on; consumers stay
unchanged. Predictions remain bit-identical to F1 at tolerance `1e-10`
(verified by the 16+12 golden snapshots).

---

## Adaptive Spectral Regularization (post-F3)

`.fit_mape_sym()` (Model 2) calls `.adaptive_spectral_shift()` from
`R/kernel-spectral.R` between the `0.5e-6` diagonal jitter step and the
SMO/osqp solve. The shift is invoked **unconditionally** (independent of
`a` or kernel type); the branch decision is made on the estimated
`λ_min(Ωs)`. This implements Theorem 2 / Algorithm 2 of
arXiv:2605.01446 v3 with one corrected detail (see "Paper deviation"
below).

### API

```
.adaptive_spectral_shift(Omega_s, T_pi = 5L, delta_stab = 1e-8) ->
  list(Omega_use, mu, lambda_min_hat, lambda_max_hat,
       branch_taken ∈ {"no_shift", "shifted"},
       n_power_iterations = c(iter1, iter2))
```

The function never modifies the input; when shifted, `Omega_use` is a
fresh matrix equal to `Omega_s + mu * I`. Results are deterministic:
both passes start from the uniform unit vector `rep(1, N) / sqrt(N)`.

### Two-pass shifted power iteration

Pass 1 runs plain power iteration on `Ωs` and returns the Rayleigh
quotient `lambda_max_hat`. Pass 2 runs power iteration on
`rho * I - Ωs` with `rho = |lambda_max_hat|` (the spectral radius)
and returns the Rayleigh `lambda_min_hat = v^T Ωs v` on the converged
`v`. Both passes are O(N²); total cost is `2 * T_pi` matvecs.

### Paper deviation

Algorithm 2 line 6 of the paper, as literally written
(`v ← -Ωs · v / ||Ωs · v||`), estimates `-λ_max(Ωs)` rather than
`λ_min(Ωs)`: power iteration on `-Ωs` converges to the eigenvector of
largest |eigenvalue|, which is `v_max(Ωs)` whenever
`|λ_max| > |λ_min|` (the typical case for Mercer-PSD kernel matrices).
The Pass 2 shift uses `|Pass 1 Rayleigh|`, not Pass 1's signed Rayleigh,
to handle the `λ_min`-dominant case where `|λ_min| > λ_max` and Pass 1
converges to `v_min` with negative Rayleigh; using `abs(...)` ensures
`rho * I - Ωs` is PSD, so Pass 2 reliably finds `v_min(Ωs)` regardless
of which side of the spectrum dominated Pass 1. This deviation is
documented inline in `R/kernel-spectral.R` and is to be flagged as a
paper-side erratum (smo-v3.tex line 3467) in F8.

### Diagnostics

`psvr_fit$solver_meta$spectral` is populated only for symmetric MAPE
fits (`loss = "mape"` and `sym != NULL`); `NULL` for Models 1, 3, 4.
Schema:

```
$mu                  numeric scalar; 0 if branch_taken == "no_shift"
$lambda_min_hat      numeric; Rayleigh from Pass 2
$lambda_max_hat      numeric; Rayleigh from Pass 1 (see note)
$branch_taken        "no_shift" | "shifted"
$n_power_iterations  integer length-2: iterations executed in Pass 1, Pass 2
```

`lambda_max_hat` reports Pass 1's signed Rayleigh and equals
`λ_max(Ωs)` for PSD or `λ_max`-dominant matrices; for the
`λ_min`-dominant pathological case it equals `λ_min(Ωs)`. The branch
decision is made correctly via `lambda_min_hat` in either case.

### Why the shifted branch is dormant in production

With the three Mercer kernels supplied by `make_kernel()` (`"rbf"`,
`"linear"`, `"polynomial"`), `Ωs = ½(Ω + a·Ω*)` is **always PSD** by
Aronszajn's closure (shift-invariant kernels) and Schur's product
theorem (polynomial kernels collapse `K - K*` for odd / even degrees
into non-negative linear combinations of valid Mercer kernels). The
shifted branch is therefore a defensive guard that activates only with
**non-Mercer kernels** (e.g., tanh / sigmoid in some parameter ranges)
supplied via a custom kernel closure — a use case not currently
documented or tested. Production fits with the supplied
`make_kernel()` types always take the no-shift branch and remain
bit-identical to F2.

### Known limitations

Theorem 2(a) guarantees that the spectrally-shifted matrix `Omega_use`
satisfies `Omega_use ⪰ delta_stab * I` in the **limit** `T_pi → ∞`.
At finite `T_pi`, the estimate of `λ_min` has a residual bias that
bleeds into `Omega_use`; the resulting matrix is "almost PSD" but may
not strictly clear the `delta_stab` floor.

For kernel matrices with well-separated spectra (the typical case for
Mercer kernels with decaying eigenvalues), `T_pi = 5` suffices for
~10⁻⁶ accuracy in `lambda_min_hat`, and `Omega_use` cleanly clears the
floor. For pathological cases with clustered spectra (e.g.,
Wigner-random matrices, where eigenvalues fill the interval
`[-rho, rho]` densely per the semicircle law), Pass 2's convergence
rate is approximately 1, and reaching the strict floor requires
`T_pi ~ 200` iterations.

Practically, even `T_pi = 20` reduces indefiniteness by ~1000× on
pathological cases, which is sufficient for the SMO solver to handle
via its existing `0.5e-6` diagonal jitter and convergence safeguards.
Callers can override the default via
`.adaptive_spectral_shift(Omega_s, T_pi = 200)`.

Note: with the three Mercer kernels supported by `make_kernel()` (RBF,
linear, polynomial), `Ωs` is always PSD (see "Why the shifted branch
is dormant" above), so the shifted branch is unreachable in production
and these limitations are not user-visible.

---

## Asymmetric Freeze + Per-pair Tolerance (post-F4)

`.smo_solve()` (`R/smo_solve.R`) now implements **Theorem 3** (asymmetric
per-sample freeze thresholds) and **Theorem 8** (per-pair tolerance
scaling) of arXiv:2605.01446 v3. Both modifications are applied
unconditionally: they reduce to the F3 defaults whenever `y` is
homogeneous (`y_k = mean(y)` for all `k`), so no opt-in flag exists.

### Theorem 3 — asymmetric freeze thresholds

The uniform shrinking threshold `n_freeze = 5L` is replaced at function
entry by two length-N integer vectors, indexed by sample type:

```
n_freeze_alpha_per[k] = max(5L, ceil(n_freeze * mean(y) / y[k]))   # for alpha_k
n_freeze_astar_per[k] = max(1L, floor(n_freeze * y[k] / mean(y)))  # for alpha*_k
```

Properties:

- Homogeneous regime (`y_k = mean(y)`): both vectors collapse to
  `n_freeze = 5L`, recovering F3 behaviour exactly.
- α*-variables tied to large `y_k` see a larger threshold (slow freeze);
  α-variables tied to small `y_k` see a larger threshold (slow freeze).
  The asymmetry exploits Lemma 4 of the paper (`α` vs `α*` saturation
  rate scales differently with `1/y_k` via the per-sample box
  `100C/y_k`).
- Convergence is preserved by the libsvm unshrinking step: any premature
  freeze is undone when the active-set gap drops to `tol`, and the
  rebuilt full `tau` is recomputed via `K_acc$get_matvec(beta)`.

### Theorem 8 — per-pair tolerance scaling

The uniform stopping tolerance `tol_eff = tol * mean(y)` is replaced **at
the convergence test only** by the per-pair value:

```
tol_pair = tol * max(y[p], y[k_j_w1])
```

where `(p, k_j_w1)` is the WSS1 convergence pair (`p` from WSS1 over
`I_up`, `k_j_w1` from the global `I_down` minimum). The convergence
test becomes `Delta = tau_i - tau_j_w1 <= tol_pair`.

The WSS3 candidate filter at the working-set selection step (`cand_mask
<- low_tau_pool < tau_i - tol_eff`) **retains the global**
`tol_eff = tol * mean(y)`. That filter is a numerical noise floor on
the candidate gap and serves a different purpose from the convergence
test; per-pair-izing it has no theorem coverage. The two scalars now
coexist:

| Variable   | Site                     | Purpose                           |
|------------|--------------------------|-----------------------------------|
| `tol_pair` | line ~131 (convergence)  | KKT gap test, scales with the WSS1 pair |
| `tol_eff`  | line ~156 (WSS3 filter)  | candidate noise floor, unchanged from F3 |

### Paper deviation (paper TODO #4)

The paper (smo-v3.tex Theorem 8) reads "j* is the WSS3-selected
variable". Implementing this literally would force WSS3 to run before
the convergence test — both wasteful and **mathematically incorrect**:
`Delta_w3 <= Delta_w1` by construction (WSS3 picks `j` to maximise
second-order gain, not minimise `tau_j`), so testing `Delta_w3` against
the tolerance would stop **prematurely**, before the true KKT
optimality gap (`= Delta_w1`) is below tolerance. The WSS1 pair
`(i_w1, j_w1)` IS the KKT optimality gap; that is the correct
convergence test. This deviation is documented inline in
`R/smo_solve.R` and flagged for a paper-side notation fix in F8 (paper
TODO #4 below).

### Empirical evidence

Snapshot fixture (`set.seed(2026); rlnorm(50, sdlog = 0.5)`,
`rho_y ~ 6.7`):
- 8 SMO-backed snapshot tests show drift `<= 8.5e-4` per prediction —
  comfortably below the per-pair tolerance floor
  `tol * max(y) ~ 2.6e-3`.
- 6 LS-SVR (Models 3, 4) snapshots stay bit-identical (no SMO).
- 6 polynomial / linear-kernel snapshots show no drift OR pre-existing
  non-convergence; see "Known issues" below.

Benchmark (`set.seed(2026); rlnorm(200, sdlog = 1.5)`,
`rho_y ~ 1273`, RBF kernel, 20 reps each):
- Heterogeneous: F3 372 iters / 0.180 s → F4 280 iters / 0.170 s. Iter
  reduction **24.7%**, wall reduction 5.6%. Iter speedup sits in the
  predicted 15–30% band (T3 ~20% × T8 ~10%, multiplicative ~32%).
- Homogeneous (`rho_y ~ 1.16`): identical iter count (10 = 10) and
  identical wall time. Default-collapse confirmed.

The wall-clock speedup is smaller than the iter speedup because at
moderate `N` the kernel-matrix construction (`O(N²·p)` work) and other
fixed overheads dominate the per-iteration cost. The wall benefit
scales with `N`.

### Default-collapse test

`tests/testthat/test-smo-solve.R` test #6 ("T3 + T8 reduce to default
behavior on homogeneous targets") fits `psvr()` on near-uniform
`y` (`rho_y ~ 1.05`) and asserts that predictions are finite and stay
positive. The `floor()`/`ceiling()` rounding in the per-sample
threshold formula may flip individual thresholds between 4, 5, and 6
when `y_k / mean(y)` crosses an integer boundary, so the trajectory may
diverge by a tiny amount; the test does not assert bit-identicality
with F3 (drift `~1e-5` is expected). RBF-kernel smoke is sufficient.

---

## Warm-start API + Working Set Selection Evaluation (post-F5)

### Working Set Selection: empirical finding

`.smo_solve()` uses Fan-Chen-Lin (2005, JMLR 6:1889–1918) WSS3 with the
unconstrained-gain score `(τ_i − τ_j)² / η`. During F5 development we
evaluated two alternatives:

- The saturation-distance multiplier proposed in arXiv:2605.01446 v3
  Theorem 4 (with tunable `alpha_wss` parameter): empirically failed on
  MAPE-SVR. With `alpha_wss = 0.5` (paper default), N=200 ρ_y=1273 RBF
  bench showed 140% iter **increase** vs unmodified WSS3. Math invariants
  held (strict descent OK, converged to same KKT optimum), but the
  multiplier penalized perfectly-good candidates whose only "fault" was
  small `R_j` relative to a mean computed over the active set. Root
  cause: the heuristic assumed room-poor candidates are a minority, but
  in MAPE-SVR's heterogeneous `C_k = 100C/y_k` regime, room-poor
  candidates dominate the active set during the early-to-mid SMO phase.

- Glasmachers-Igel (2006, JMLR 7:1437–1466) maximum-gain WSS3 with exact
  box-clipped realized gain: empirically equivalent to WSS3 on tested
  regimes (N=50/200, ρ_y=6.7–1273, RBF). Iter counts matched
  bit-exactly (137=137, 202=202, 43=43 snapshot fixture; 280=280 bench).
  Max-gain incurred ~12% per-iteration wall overhead from the additional
  clipping arithmetic. Root cause for the equivalence: WSS3's analytic
  step `δ_unc = gap / η` typically fits within the per-sample box bounds
  even for highly heterogeneous `C_k`, so clipping rarely activates and
  max-gain reduces to WSS3 numerically.

Conclusion: the "saturation problem" both the heuristic and max-gain
were attempting to solve is a phantom at the WSS3 selection level for
MAPE-SVR. Fan-Chen-Lin's standard libsvm WSS3 is sufficient. This is a
useful negative result for future MAPE-SVR optimization research. See
paper TODO #5 for the implication on smo-v3.tex Theorem 4.

### Warm-start API (Theorem 5 with paper deviation)

The package provides direct warm-start via the `alpha_init` and
`alpha_star_init` parameters of `psvr()` (validated to length N,
strictly positive targets for MAPE loss). The `.warm_start_init()`
helper in `R/warm_start.R` implements Algorithm 1 of arXiv:2605.01446
v3 with one deviation from the paper text:

- **Paper Algorithm 1 Step 2:** uniform shift over ALL `N` samples by
  `violation / N`. Empirically degrades the warm-start advantage on
  10-fold CV (Round 1: 0.97× cumulative).
- **Our deviation:** distribute violation only over the new-sample
  subset (`S_new \ S_prev`). Rationale: the equality-constraint
  violation arises entirely from removed samples (`S_prev \ S_new`)
  whose dual values are no longer used; retained samples
  (`S_prev ∩ S_new`) were at the equality-constraint manifold at the
  previous fold's optimum and should be preserved exactly. The
  new-sample-only projection preserves retained values to `1e-12`
  (test verified). When the per-new shift forces clipping at `0` or
  `C_k`, a one-pass uniform refinement absorbs the residual (rare in
  typical CV).

This deviation is documented as paper TODO #6 for incorporation into
the final paper text.

### CV helper `psvr_cv()`

`psvr_cv(splits, X_var, y_var, ...)` accepts an `rsample::rset` OR a
list-of-tuples and orchestrates warm-start across folds using
row-ID-based `new_mask` inference. Returns a plain tibble with
`split_id`, `fit`, `predictions`, `metrics`, `iter_count`,
`elapsed_sec`, `warm_started`.

Scope: A′ (`psvr_cv` as explicit helper, no parsnip auto-warm-start in
`tune_grid`). F5b will add parsnip integration if/when warranted.

### Empirical speedup calibration

Paper-predicted cumulative speedup: **3–7×** on 10-fold CV with
linear-convergence assumption (`T_warm / T_cold ≈ 0.2` per fold).

Observed:

- N=300 (ρ_y=2388, RBF, 10-fold): **1.12× wall**, 12.7% iter reduction.
- N=1000 (ρ_y=16265, RBF, 10-fold): **1.14× wall**, 13.8% iter reduction.

Per-fold `T_warm / T_cold ≈ 0.88` (not 0.2). Cumulative speedup is
approximately linear in fold count, not exponential. This is
N-independent at our tested regimes — see paper TODO #7 for the
recalibration recommendation.

### Breaking-change: `fit$alpha` → `fit$beta` for MAPE

For MAPE fits (`psvr_fit` with `loss = "mape"`), the field formerly
called `fit$alpha` (length `n_sv`, post-pruning, holding `β = α − α*`)
is renamed to `fit$beta`. Two new length-`N` (pre-pruning) fields
`fit$alpha` and `fit$alpha_star` expose the true SMO dual variables —
required as warm-start state by `psvr_cv()`. LS-SVR fits
(`loss = "rmspe"`) retain previous semantics: `fit$alpha` is the
linear-system solution, `fit$alpha_star = NULL`, `fit$beta = NULL`.
Downstream code reading `fit$alpha` from MAPE fits must switch to
`fit$beta`.

`psvr_fit$solver_meta` now propagates real `iters` and `converged`
values from the SMO solver (previously hard-coded to `NA`).

---

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

---

## Known issues

### TODO #5 — Pre-existing SMO convergence pathology on linear / polynomial kernels

`.smo_solve()` fails to converge within `max_iter = 100000` on MAPE
fits with linear and polynomial kernels (RBF works correctly). Symptom:
`solver_meta$converged = FALSE`, `iterations = max_iter`. Tests
document this via "did not converge" warnings in
`test-bit-identical.R` and `test-psvr-direct.R` for `mape_lin`,
`mape_poly`, `mape_sym_poly`. The pathology predates F1 and was
identified during F4 drift quantification (6 of 28 snapshot tests show
the `STALE F3+F4` verdict — both F3 and F4 stall at `max_iter` at
divergent non-converged endpoints).

Root cause unknown — candidates: (a) kernel-matrix near-degeneracy on
the synthetic test data; (b) bias-update instability with non-RBF
kernels; (c) shrinking heuristic mis-fires on linear-kernel `tau`
distributions.

Defer investigation; not blocking the F1–F8 refactor track. Production
use of `psvr()` with linear / polynomial kernels and MAPE loss should
be reviewed before relying on results — prefer the `osqp` backend
(`solver = "osqp"`), which has its own internal convergence guard and
does not exhibit this pathology, when accuracy is critical.

---

## Paper TODOs (to be applied in F8)

| # | Location | Issue | Severity |
|---|---|---|---|
| 1 | smo-v3.tex:3467 Algorithm 2 line 6 | `v ← -Ωs · v / ‖Ωs · v‖` estimates `−λ_max(Ωs)`, not `λ_min(Ωs)`. Fix: two-pass shifted power iteration with `abs(Pass 1 Rayleigh)` as the Pass 2 shift. Implementation in `R/kernel-spectral.R`; deviation documented inline. | Algorithmic bug |
| 2 | smo-v3.tex:3525–3536 | Claim that polynomial degree 2 + a = −1 yields generically indefinite `Ωs` is algebraically wrong (Schur product theorem: all Mercer kernels yield PSD `Ωs`). Shifted branch is dormant in production with `make_kernel()` Mercer kernels. | Algebraic error |
| 3 | smo-v3.tex:3448–3454 | "`T_pi = 3–5` suffices for `δ = 10⁻⁶`" assumes well-separated spectrum. For clustered spectra (Wigner), requires `T_pi ~ 200+`. | Overly optimistic |
| 4 | smo-v3.tex:4173–4187 Theorem 8 | "`j*` is the WSS3-selected variable" should read "convergence-pair variable (WSS1 `j*`)". WSS3 `j*` in the convergence test would break optimality (`Δ_w3 ≤ Δ_w1`). Notation slip, mathematically incorrect as written. | Notation slip |
| 5 | smo-v3.tex Section 6 / Theorem 4 | **Drop Theorem 4 from the paper.** The proposed saturation-distance multiplier empirically fails (140% iter increase on N=200 ρ_y=1273 RBF). The "saturation problem" is a phantom: WSS3's `δ_unc` typically fits the per-sample box, so neither the heuristic nor Glasmachers-Igel max-gain provides empirical benefit on MAPE-SVR. Fan-Chen-Lin WSS3 (libsvm) is sufficient. Recompute Corollary 6 without T4's 1.15× multiplier. | Falsified novel claim; substituted by negative empirical finding |
| 6 | smo-v3.tex Algorithm 1 Step 2 | Replace uniform shift over `N` with new-samples-only projection. Rationale: violation arises entirely from removed samples; retained samples were at the equality-constraint manifold at the previous fold's optimum. Mathematically valid (post-Step-2 violation = 0, convergence preserved); empirically gains 0.17× cumulative speedup on 10-fold CV (0.97× → 1.14×). | Paper-text deviation with positive empirical impact |
| 7 | smo-v3.tex Theorem 5 speedup prediction | Predicted 3–7× cumulative on 10-fold CV is over-optimistic. Empirical observation at N=300 (ρ_y=2388) and N=1000 (ρ_y=16265): both ~1.12–1.14× cumulative (N-independent). Per-fold `T_warm / T_cold ≈ 0.88`, not 0.2. Cumulative speedup is approximately linear in fold count, not exponential. Recalibrate the prediction or restrict applicable regime. | Empirical calibration |
| 8 | smo-v3.tex Theorem 6 (LIBSVM column cache) | Architectural mismatch with psvr's full-matrix design. The proposed column-on-demand LRU cache is designed for an SMO that synthesizes columns of Ω per iteration; psvr's `.make_kernel_accessor()` (F2) is a thin read-only closure over an already-materialised matrix, so T6 saves no memory in the dominant `kernel_matrix(K, X)` pre-construction path. F6 instead Rcpp-accelerates the `O(N²·p)` construction step (~12× wall) and adds cross-fold reuse in `psvr_cv()`. Either drop T6 from the paper or restructure the paper architecture description to match the implementation (full-matrix, no LRU cache). | Architectural mismatch; substituted by negative finding and Rcpp acceleration |

---

## R environment

- **R 4.5.3** at `C:\Program Files\R\R-4.5.3\bin\Rscript.exe`
- Run from PowerShell: `& "C:\Program Files\R\R-4.5.3\bin\Rscript.exe" script.R`
- Always use forward slashes or single-quoted strings in PowerShell to avoid `\U` Unicode escape errors
- **osqp 1.0.0** uses `solve_osqp(P, q, A, l, u, pars)` directly — not `osqp(...)$solve()` (S7 object, old R6 API is gone)
- Load package in R session: `devtools::load_all("C:/Users/behep/OneDrive - ITESO/PhD/00-Tesis/psvr")`

---

## Invariants to enforce everywhere

- `yk > 0` always — validated at fit time, never silently coerced.
- `a ∈ {-1, 1}` for symmetric models.
- The symmetric kernel `Ks(xi, xj) = K(xi, xj) + a·K(xi, -xj)` uses negation of `xj`,
  so the kernel closure must accept negative inputs even when training data is positive.
- `YΓ` is diagonal — add to `diag(Omega)` in place, never build an N×N diagonal matrix.
- Box constraints in Models 1 & 2 are per-sample: `|βk| ≤ 100C/yk`.
- Symmetric models (2 and 4) build `Ωs = ½(Ω + a·Ω*)` via
  `sym_kernel_matrix()` and pass `Ωs` directly to the solver — no extra
  `0.5 *` at the call site. Predictions use `sym_kernel_vector()`, which
  already returns `½ Ks(xk, x)`.
