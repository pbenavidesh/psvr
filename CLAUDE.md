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
15. [x] **F7** — Theorem 7 from arXiv:2605.01446 v3: block-k=4 SMO with
        descent-guaranteed decoupling. Each outer SMO iteration may
        select a second pair `(i_2, j_2)` and apply a 2-D joint update
        when the descent criterion holds. Strict sample-level
        disjointness; corrected sign-free xi formula. R-level
        implementation reduces iter count 38–48% on converging
        regimes but has 2× per-iter overhead → wall neutral-to-
        negative in R. See sections "Block-k=4 SMO with descent-
        guaranteed decoupling (post-F7)" and "Portable C++
        architecture (post-F7-C-full)".
16. [x] **F7 C-full portable** — port SMO inner loop + block-k=4 +
        F6 kernels to a portable C++ core (`src/core_*.cpp`, pure
        std-library types) with a thin Rcpp binding layer
        (`src/binding_*.cpp`). New `engine = c("rcpp", "r")`
        parameter on `psvr()`; default `"rcpp"`. `engine = "r"` is
        the bit-identical reference, preserved through v0.0.3.x;
        deprecated v0.0.4.0; removal v0.1.0. Bit-identical across
        16 configs (Models × 4 kernels × 2 modes). F4 baseline
        1.60–4.69× wall speedup; F7 path 1.94–7.73×. F7 vs F4 wall:
        +12.2% R1, +17.5% R4 — paper TODO #9 RESOLVED on converging
        regimes. CV B3-rcpp 4.28× over the F4+F5-R baseline.
17. [ ] **F8** — Paper-side errata fixes (#1–#11).

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
SMO/osqp solve — a two-pass shifted power iteration that estimates
`λ_min(Ωs)` and adds `μ·I` only if indefinite. With the three Mercer
kernels supplied by `make_kernel()` the shifted branch is dormant in
production; diagnostics surface under `psvr_fit$solver_meta$spectral`
for symmetric MAPE fits only. Paper Algorithm 2 line 6 has a sign bug
(paper TODO #1).

See [docs/archive/F3-spectral-regularization.md](docs/archive/F3-spectral-regularization.md)
for the API, two-pass algorithm, paper-deviation derivation, diagnostics
schema, and finite-`T_pi` limitations.

---

## Asymmetric Freeze + Per-pair Tolerance (post-F4)

`.smo_solve()` (`R/smo_solve.R`) implements Theorem 3 (asymmetric
per-sample freeze thresholds, `n_freeze_alpha_per`/`n_freeze_astar_per`
vectors derived from `mean(y) / y[k]`) and Theorem 8 (per-pair
tolerance scaling `tol_pair = tol * max(y[p], y[k_j_w1])` at the
convergence test only — the WSS3 candidate filter keeps the global
`tol_eff`). Both apply unconditionally and collapse to F3 defaults on
homogeneous targets. The paper text for T8 says "WSS3-selected `j*`" but
must read "WSS1 convergence-pair `j*`" — flagged as paper TODO #4.
Empirical: ~24.7% iter reduction at N=200 ρ_y=1273 RBF.

See [docs/archive/F4-freeze-tolerance.md](docs/archive/F4-freeze-tolerance.md)
for the per-sample formulas, the `tol_pair` vs `tol_eff` site table, the
paper-deviation argument, and the benchmark numbers.

---

## Warm-start API + Working Set Selection (post-F5)

`psvr()` accepts `alpha_init` and `alpha_star_init`; `.warm_start_init()`
(`R/warm_start.R`) implements Algorithm 1 of arXiv:2605.01446 v3 with a
deviation — violation distributed over `S_new \ S_prev` only, not
uniformly over `N`. `psvr_cv()` (`R/psvr_cv.R`) orchestrates warm-start
across folds using row-ID-based `new_mask`. Empirical cumulative
speedup on 10-fold CV is **1.12–1.14×** (paper predicted 3–7×;
recalibration is paper TODO #7). WSS evaluation: Fan-Chen-Lin libsvm
WSS3 is sufficient — both the paper's saturation-distance multiplier
(Theorem 4) and Glasmachers-Igel max-gain showed no empirical benefit
on MAPE-SVR; T4 should be dropped from the paper (paper TODO #5).

**Breaking change:** for MAPE fits `fit$alpha` was renamed to
`fit$beta`; new length-N `fit$alpha` / `fit$alpha_star` expose the SMO
dual variables for warm-start.

See [docs/archive/F5-warmstart-wss.md](docs/archive/F5-warmstart-wss.md)
for the Algorithm 1 deviation argument, the WSS3 alternative evaluation,
and the speedup-calibration evidence.

---

## Rcpp-accelerated kernel construction (post-F6)

Three Rcpp kernels (`src/kernel_rbf.cpp`, `kernel_linear.cpp`,
`kernel_poly.cpp`) replace the R nested loop in `kernel_matrix()` via
attribute-based dispatch (`attr(K, "kernel_info")` set by
`make_kernel()`). `psvr_cv()` computes `Omega_full` once across the
union of all folds' `in_id` and slices per fold. Internal
`precomputed_Omega` / `precomputed_Omega_s` channel bypasses per-call
kernel build. Bit-identical with legacy R via three FP-discipline
choices: long-double accumulators, `powl` for polynomial `degree >= 3`,
and no self-kernel symmetry shortcut. Wall: ~12× on construction.
Paper T6 (LIBSVM column-on-demand cache) is skipped — architectural
mismatch with the full-matrix design (paper TODO #8).

See [docs/archive/F6-rcpp-kernels.md](docs/archive/F6-rcpp-kernels.md)
for the bit-identicality discipline, `psvr_cv()` reuse mechanics,
performance characteristics, and the T6-skip rationale.

---

## Block-k=4 SMO (post-F7)

Theorem 7 of arXiv:2605.01446 v3: after WSS1+WSS3 select pair 1
`(i_1, j_1)`, optionally select a sample-disjoint pair 2 `(i_2, j_2)`
and apply a 2-D joint update governed by a descent-guaranteed
decoupling test. Falls back to the standard 1-D update otherwise.
Key correction vs the paper spec: the xi cross-curvature formula is
**sign-free** under MAPE-SVR's α/α* dual (`xi = K_p[p_2] - K_p[q_2] -
K_q[p_2] + K_q[q_2]`, no `s_i s_j` factors). Pair-2 selection uses
`alpha_couple = 0.5` coupling-penalty default. Default-collapse via
`block_k4_enabled = FALSE` is bit-identical to F4 on both engines.
Iter reduction 38–48% on converging regimes. T5 × T7 do NOT compose
multiplicatively in CV (paper TODO #10). Telemetry:
`solver_meta$joint_updates`, `k2_fallbacks`, `decoupling_rate`,
`early_phase_*`, `late_phase_*`.

See [docs/archive/F7-block-k4.md](docs/archive/F7-block-k4.md) for the
Δβ invariant derivation, the descent test (D1), pair-2 selection (D2),
per-regime bench tables, and the T5 × T7 B-suite results.

---

## Portable C++ architecture (post-F7-C-full)

The SMO inner loop, block-k=4 logic, and F6 kernels are ported to a
portable C++ core (`src/core_*.cpp`, std-library types only) with a
thin Rcpp binding (`src/binding_*.cpp`). `psvr()` exposes
`engine = c("rcpp", "r")` (default `"rcpp"`). The R path is preserved
as the bit-identical reference under the engine selector. Per-iter
overhead drops from 2.0× (R) to 1.40× (Rcpp), restoring T7 wall
positivity (+12.2% R1, +17.5% R4 vs F4 baseline) — paper TODO #9
resolved on converging regimes. CV B3-rcpp achieves 4.28× over the
F4+F5-R baseline.

Bit-identicality discipline (loop direction, `which.max` strict-`>`
tie-break, long-double accumulators, separate-not-fused τ subtractions)
is locked by `tests/testthat/test-engine-equivalence.R` (16-config
canary). Conditional compile pattern (`PSVR_STANDALONE_BUILD` /
`<R_ext/BLAS.h>`) demonstrates the portability claim (paper TODO #11).

**Build invariant — do not change:** `src/Makevars` and `Makevars.win`
specify `PKG_LIBS = $(BLAS_LIBS) $(FLIBS)`. `core_smo_solve.cpp` calls
`F77_CALL(dgemv)`; removing `$(BLAS_LIBS)` breaks linking. Do not add
`$(LAPACK_LIBS)` (no LAPACK dependency); if ever needed, order must be
`$(LAPACK_LIBS) $(BLAS_LIBS) $(FLIBS)` per *Writing R Extensions* §1.2.3.

See [docs/archive/F7-C-full-portable.md](docs/archive/F7-C-full-portable.md)
for the file-by-file architecture, memory layout & ownership rules,
BLAS access pattern, full bit-identicality discipline notes, and the
build-system rationale.

---

## engine = "r" lifecycle

`engine = c("rcpp", "r")` is the engine selector on `psvr()` and
`.smo_solve()`. `"rcpp"` is the default; `"r"` dispatches to
`.smo_solve_r()` (the original R algorithm, renamed in F7-C-full
and preserved as the bit-identical reference).

### Graduation timeline

- **v0.0.2.9006 – v0.0.3.x** (CURRENT): both engines live and
  numerically equivalent. `engine = "rcpp"` default; `engine = "r"`
  available for debugging / teaching / canary comparison. All
  equivalence tests cover both engines.
- **v0.0.4.0**: emit `lifecycle::deprecate_soft("0.0.4.0",
  ".smo_solve(engine = 'r')")` on `engine = "r"` invocations.
- **v0.1.0**: remove `engine = "r"` entirely. Delete
  `.smo_solve_r()`; `R/smo_solve.R` becomes pure dispatch (or is
  inlined into the binding layer).

### Graduation criteria for v0.1.0 removal

All three must hold:

1. v0.1.0 prep is underway (concrete release branch open).
2. Snapshot tests pass with `engine = "rcpp"` for **at least 2
   release cycles** without bugs caught by the R-equivalence canary
   (`test-engine-equivalence.R`).
3. Bench suite (`dev/bench-F7.R`) covers all paper regimes under
   `engine = "rcpp"` (currently: R1-R4 plus T5 × T7 stacking).

The 2-release-cycle requirement is the load-bearing one: it ensures
the C++ port has accumulated enough field exposure to surface any
hidden bugs that the synthetic test fixtures missed.

### Why we kept the dual engine instead of removing R immediately

Risk mitigation. The C++ port is mechanical (~1200 lines) but
intricate (bit-identicality requires loop-order discipline, tie-
break preservation, FP-associativity in the fused τ update). The R
reference catches algorithmic differences that snapshot tests alone
might miss — e.g., the Phase 2 STOP 2 8.88e-16 drift would have
shipped silently if the canary tests didn't compare engine-vs-
engine on every model × kernel × mode combination.

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
| 9 | smo-v3.tex Theorem 7 wall-time claim | **RESOLVED via C++ port (F7-C-full)**. T7's per-iteration overhead is *implementation-dependent*. R-level: 2.0× per-iter overhead → net wall regression (−25.6% R1, +2.4% R4). C++ binding: 1.40× per-iter overhead → net wall positive (+12.2% R1, +17.5% R4). The paper's "2× more progress per iteration" claim is correct theoretically and implementation-independent; wall translation requires accounting for language-level scaffolding overhead. **Recommendation**: paper should report iter speedup (theoretically validated, language-agnostic) and wall speedup (implementation-dependent) SEPARATELY. The C++ implementation restores wall positivity, validating Theorem 7's practical utility on converging regimes. Also flag: T7 cannot help on regimes where the baseline does not converge (R2/R3 hit max_iter; paper TODO #5). | Implementation-dependent; resolved via portable C++ core |
| 10 | smo-v3.tex Corollary 6 (T5 × T7 cumulative speedup) | T5 (warm-start) and T7 (block-k=4) **do not compose multiplicatively** in CV. B2 (T7 alone) achieves 23 363 iters and 0.508 s wall in 10-fold CV at N=300, ρ_y=2388; B3 (T7+warm-start) achieves 26 586 iters (+13.8%) and 0.515 s. Confirmed under both engines: B2-r 2.697 s vs B3-r 2.489 s; B2-rcpp 0.508 s vs B3-rcpp 0.515 s. The non-multiplicativity is **algorithmic** (T5 perturbation cost on top of T7's shortened trajectory), not implementation-specific. **Recommendation**: paper should restructure Corollary 6 cumulative speedup as `max(T5, T7)` in CV regimes, not `T5 × T7`. The warm-start projection step (Algorithm 1) takes constant overhead per fold; T7 dominates the per-fold iter reduction; combined they sit at T7's level with a small projection overhead. | Algorithmic finding; both engines validate |
| 11 | smo-v3.tex Section on implementation / language portability | **NEW.** The C++ core (`src/core_*.cpp`) is independent of R types (no `Rcpp::*`, no R API calls outside the optional `<R_ext/BLAS.h>` for dgemv). The R binding (`src/binding_*.cpp`, ~180 lines total) is thin and uses Rcpp idiomatically. Future pybind11 binding for Python would follow the same pattern, wrapping the same core with `PSVR_PYTHON_BUILD` define for numpy-backed BLAS hookup. The conditional compilation pattern is demonstrated in `core_smo_solve.cpp` (PSVR_STANDALONE_BUILD path used by `dev/check_core.cpp`). **Recommendation**: paper should add a brief "Implementation portability" subsection citing the architecture as evidence that SVR algorithms with intricate FP-discipline requirements (bit-identicality, tie-break preservation, FP-associativity ordering) can be cleanly separated from host-language idioms via a flat-buffer C++ API. This validates the paper's portability claim with a concrete demonstrated implementation. | New finding; concrete implementation demonstration |

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
