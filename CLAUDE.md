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
- **S3 classes:** each model returns a named list with class:
  - `"psvr_mape"` (Model 1)
  - `"psvr_mape_sym"` (Model 2)
  - `"psvr_rmspe"` (Model 3)
  - `"psvr_rmspe_sym"` (Model 4)
- **S3 methods:** `predict.psvr_*()` for each class.
- **Documentation:** roxygen2 with `@param`, `@return`, `@examples`.
- **Style:** tidyverse / base R style: `snake_case`, `<-` assignment.
- **License:** MIT.

---

## Implementation order

1. [x] `CLAUDE.md` — this file
2. [x] Package scaffolding: `DESCRIPTION`, `NAMESPACE`, directory structure
3. [x] `R/kernel.R` — `make_kernel()` helper
4. [x] `R/rmspe_lssvr.R` — Model 3: `rmspe_lssvr()` + `predict.psvr_rmspe()`
5. [x] `R/rmspe_sym_lssvr.R` — Model 4: `rmspe_sym_lssvr()` + `predict.psvr_rmspe_sym()`
6. [x] `R/mape_svr.R` — Model 1: `mape_svr()` + `predict.psvr_mape()`
7. [x] `R/mape_sym_svr.R` — Model 2: `mape_sym_svr()` + `predict.psvr_mape_sym()`
8. [x] `tests/` — testthat unit tests for all four models (84 tests, 0 failures)

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
- Prediction for Model 2: multiply by `½` (from symmetric representer theorem).
- Prediction for Model 4: use `½(K(xk,x) + a·K(xk,-x))` per support vector.
