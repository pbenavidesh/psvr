# psvr — Percentage-error Support Vector Regression

## Package purpose

`psvr` implements four SVR models derived from a unified mathematical
framework for percentage-error loss functions. It targets forecasting
contexts where targets `yk > 0` are strictly positive and relative
accuracy matters more than absolute accuracy.

Source paper: *A Unified Family of Percentage-Error Support Vector
Regression Models with Symmetric Kernel Extensions* (MDPI Mathematics,
2026).

## Related repositories

| Repo | Role | Rule |
|----|----|----|
| `psvr` | This R package | Active development |
| `psvr-paper` | Frozen reproducibility artifact for the published paper | **Never modify** |
| `smo-paper` | SMO-efficiency preprint (arXiv:2605.01446) | Consumes `psvr` internals |
| `one-ring` | Thesis source | Independent track |

Do not carry content between repos without an explicit instruction to do
so.

## Mathematical notation

| Symbol | Meaning |
|----|----|
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

## The four models

| \# | Model | `loss` | `sym` | Solver | Key file |
|----|----|----|----|----|----|
| 1 | ε-SVR with MAPE | `"mape"` | `NULL` | SMO (default) or osqp | `R/mape_svr.R` |
| 2 | Symmetric ε-SVR with MAPE | `"mape"` | `±1L` | SMO (default) or osqp | `R/mape_sym_svr.R` |
| 3 | LS-SVR with RMSPE | `"rmspe"` | `NULL` | [`base::solve()`](https://rdrr.io/r/base/solve.html) | `R/rmspe_lssvr.R` |
| 4 | Symmetric LS-SVR with RMSPE | `"rmspe"` | `±1L` | [`base::solve()`](https://rdrr.io/r/base/solve.html) | `R/rmspe_sym_lssvr.R` |

Models 1–2 solve a dual QP with the sample-dependent box
`|βk| ≤ 100C/yk` and `Σk βk = 0`. Models 3–4 solve the `(N+1)×(N+1)`
augmented linear system with `YΓ` added to the diagonal of `Ω` (Model 3)
or `Ωs` (Model 4).

### Kernel symmetry (Assumption 3)

Models 2 and 4 require `K(-xi, xj) = K(xi, -xj)` and
`K(-xi, -xj) = K(xi, xj)`. The RBF kernel and polynomial kernels of even
degree satisfy both.

## Architecture

- **`psvr(X, y, loss, sym, kernel, ...)`** is the single entry point;
  returns a `psvr_fit` object (`R/psvr-main.R`). Methods:
  [`predict()`](https://rdrr.io/r/stats/predict.html),
  [`print()`](https://rdrr.io/r/base/print.html),
  [`coef()`](https://rdrr.io/r/stats/coef.html),
  [`summary()`](https://rdrr.io/r/base/summary.html).
- **[`psvr_cv()`](https://pbenavidesh.github.io/psvr/reference/psvr_cv.md)**
  (`R/psvr_cv.R`) — fold-wise fitting for `loss = "mape"` with
  warm-start carryover.
- **6 parsnip specs**
  ([`psvr_mape_rbf()`](https://pbenavidesh.github.io/psvr/reference/psvr_mape_specs.md)
  etc.) plus dials helpers. Symmetry is the `sym_type` argument
  (`"none"` / `"even"` / `"odd"`), not a separate spec.
- **The four legacy wrappers are GONE.** `mape_svr()`, `mape_sym_svr()`,
  `rmspe_lssvr()`, `rmspe_sym_lssvr()` and `R/deprecated.R` were removed
  in 0.0.2.9010. Their four fit **classes** and all 20 methods remain —
  they are what the parsnip engine fit wrappers return.
- **Internal helpers**:
  [`.fit_mape()`](https://pbenavidesh.github.io/psvr/reference/dot-fit_mape.md),
  [`.fit_mape_sym()`](https://pbenavidesh.github.io/psvr/reference/dot-fit_mape_sym.md),
  [`.fit_rmspe()`](https://pbenavidesh.github.io/psvr/reference/dot-fit_rmspe.md),
  [`.fit_rmspe_sym()`](https://pbenavidesh.github.io/psvr/reference/dot-fit_rmspe_sym.md);
  validation in `R/utils-validation.R`; predict dispatch in
  `R/utils-predict.R`.
- **Kernel interface**: `make_kernel(type, ...)` returns a closure
  carrying `attr(K, "kernel_info")`;
  [`kernel_matrix()`](https://pbenavidesh.github.io/psvr/reference/kernel_matrix.md)
  dispatches through it to the Rcpp implementations in
  `src/kernel_*.cpp`. Closures without the attribute fall back to the
  R-only
  [`.legacy_kernel_matrix()`](https://pbenavidesh.github.io/psvr/reference/dot-legacy_kernel_matrix.md).
- **Engine selector**: `engine = c("rcpp", "r")` on
  [`psvr()`](https://pbenavidesh.github.io/psvr/reference/psvr.md),
  default `"rcpp"`. `"r"` is the bit-identical reference implementation,
  preserved through v0.0.3.x, soft-deprecated at v0.0.4.0, removed at
  v0.1.0.

## Invariants to enforce everywhere

- `yk > 0` always — validated at fit time, never silently coerced.
- `a ∈ {-1, 1}` for symmetric models.
- `Ks(xi, xj)` negates `xj`, so kernel closures must accept negative
  inputs even when training data is strictly positive.
- `YΓ` is diagonal — add to `diag(Omega)` in place, never build an N×N
  diagonal matrix.
- Box constraints in Models 1 and 2 are per-sample: `|βk| ≤ 100C/yk`.
- Symmetric models build `Ωs = ½(Ω + a·Ω*)` via
  [`sym_kernel_matrix()`](https://pbenavidesh.github.io/psvr/reference/sym_kernel_matrix.md)
  and pass `Ωs` straight to the solver — no extra `0.5 *` at the call
  site. Predictions use
  [`sym_kernel_vector()`](https://pbenavidesh.github.io/psvr/reference/sym_kernel_vector.md),
  which already returns `½ Ks(xk, x)`.

## Conventions

- **Style**: tidyverse / base R — `snake_case`, `<-` assignment.
- **Docs**: roxygen2 with `@param`, `@return`, `@examples`.
- **Target validation**: every fitter calls `stopifnot(all(y > 0))`.
- **License**: MIT.

## Build and environment

- **R 4.5.3** at `C:\Program Files\R\R-4.5.3\bin\Rscript.exe`.
- From PowerShell:
  `& "C:\Program Files\R\R-4.5.3\bin\Rscript.exe" script.R`. Use forward
  slashes or single-quoted strings to avoid `\U` Unicode escape errors.
- Load in a session with `devtools::load_all()` from the repo root.
- **osqp 1.0.0** uses `solve_osqp(P, q, A, l, u, pars)` directly — the
  old R6 `osqp(...)$solve()` API is gone.
- **`README.md` is regenerated with `rmarkdown::render("README.Rmd")`,
  NEVER [`knitr::knit()`](https://rdrr.io/pkg/knitr/man/knit.html).**
  `README.Rmd` declares `output: github_document`, which is a pandoc
  writer: `render()` applies it, `knit()` skips pandoc entirely. A raw
  knit leaks the YAML front matter into `README.md`, re-wraps prose to
  ~85 columns instead of ~72, strips the alignment padding from tables,
  and turns ```` ``` r ```` fences into ```` ```r ````. Nothing guards
  this — the pre-commit hook checks that `README.md` is not older than
  `README.Rmd` and that both are staged, not which renderer produced it.
  After rendering, diff every printed value: the Quick-start numbers
  must reproduce byte-for-byte unless `R/psvr-main.R` changed.
- **Build invariant — do not change**: `src/Makevars` and `Makevars.win`
  set `PKG_LIBS = $(BLAS_LIBS) $(FLIBS)`. `core_smo_solve.cpp` calls
  `F77_CALL(dgemv)`, so removing `$(BLAS_LIBS)` breaks linking. Do not
  add `$(LAPACK_LIBS)`; if ever needed the order must be
  `$(LAPACK_LIBS) $(BLAS_LIBS) $(FLIBS)` per *Writing R Extensions*
  §1.2.3.

## Known limitation

The built-in SMO solver does not converge within `max_iter` on MAPE fits
with **linear and polynomial** kernels (RBF is unaffected);
`solver_meta$converged` is `FALSE` and `iterations` hits the cap.
Several tests document this with “did not converge” warnings. Root cause
is not yet established. Prefer `solver = "osqp"` for those kernel/loss
combinations when accuracy matters.

------------------------------------------------------------------------

@.claude/CLAUDE.md
