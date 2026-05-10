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
  type ∈ `{"rbf", "linear", "polynomial"}`. Shared across all four models.
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
11. [ ] **F3** — Algorithm 2: adaptive spectral shift (placeholder in `R/kernel-spectral.R`).
12. [ ] **F4–F8** — Theorems 3–8 from arXiv:2605.01446 v3.

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
