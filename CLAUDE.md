# psvr — Percentage-error Support Vector Regression

## Package purpose

`psvr` implements four SVR models derived from a unified mathematical framework for
percentage-error loss functions. It targets forecasting contexts where targets
`yk > 0` are strictly positive and relative accuracy matters more than absolute
accuracy.

Source paper: *A Unified Family of Percentage-Error Support Vector Regression Models
with Symmetric Kernel Extensions* (MDPI Mathematics, 2026).

## Related repositories

| Repo | Role | Rule |
|---|---|---|
| `psvr` | This R package | Active development |
| `psvr-paper` | Frozen reproducibility artifact for the published paper | **Never modify** |
| `smo-paper` | SMO-efficiency preprint (arXiv:2605.01446) | Consumes `psvr` internals |
| `one-ring` | Thesis source | Independent track |

Do not carry content between repos without an explicit instruction to do so.

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

## The four models

| # | Model | `loss` | `sym` | Solver | Key file |
|---|---|---|---|---|---|
| 1 | ε-SVR with MAPE | `"mape"` | `NULL` | SMO (default) or osqp | `R/mape_svr.R` |
| 2 | Symmetric ε-SVR with MAPE | `"mape"` | `±1L` | SMO (default) or osqp | `R/mape_sym_svr.R` |
| 3 | LS-SVR with RMSPE | `"rmspe"` | `NULL` | `base::solve()` | `R/rmspe_lssvr.R` |
| 4 | Symmetric LS-SVR with RMSPE | `"rmspe"` | `±1L` | `base::solve()` | `R/rmspe_sym_lssvr.R` |

Models 1–2 solve a dual QP with the sample-dependent box `|βk| ≤ 100C/yk` and
`Σk βk = 0`. Models 3–4 solve the `(N+1)×(N+1)` augmented linear system with `YΓ`
added to the diagonal of `Ω` (Model 3) or `Ωs` (Model 4).

### Kernel symmetry (Assumption 3)

Models 2 and 4 require `K(-xi, xj) = K(xi, -xj)` and `K(-xi, -xj) = K(xi, xj)`.
The RBF kernel and polynomial kernels of even degree satisfy both.

## Architecture

- **`psvr(X, y, loss, sym, kernel, ...)`** is the single entry point; returns a
  `psvr_fit` object (`R/psvr-main.R`). Methods: `predict()`, `print()`, `coef()`,
  `summary()`.
- **`psvr_cv()`** (`R/psvr_cv.R`) — fold-wise fitting for `loss = "mape"` with
  warm-start carryover.
- **6 parsnip specs** (`psvr_mape_rbf()` etc.) plus dials helpers. Symmetry is the
  `sym_type` argument (`"none"` / `"even"` / `"odd"`), not a separate spec.
- **The four legacy wrappers are GONE.** `mape_svr()`, `mape_sym_svr()`,
  `rmspe_lssvr()`, `rmspe_sym_lssvr()` and `R/deprecated.R` were removed in
  0.0.2.9010. Their four fit **classes** and all 20 methods remain — they are what
  the parsnip engine fit wrappers return.
- **Internal helpers**: `.fit_mape()`, `.fit_mape_sym()`, `.fit_rmspe()`,
  `.fit_rmspe_sym()`; validation in `R/utils-validation.R`; predict dispatch in
  `R/utils-predict.R`.
- **Kernel interface**: `make_kernel(type, ...)` returns a closure carrying
  `attr(K, "kernel_info")`; `kernel_matrix()` dispatches through it to the Rcpp
  implementations in `src/kernel_*.cpp`. Closures without the attribute fall back to
  the R-only `.legacy_kernel_matrix()`.
- **Engine selector**: `engine = c("rcpp", "r")` on `psvr()`, default `"rcpp"`.
  `"r"` is the bit-identical reference implementation, preserved through v0.0.3.x,
  soft-deprecated at v0.0.4.0, removed at v0.1.0.

## Invariants to enforce everywhere

- `yk > 0` always — validated at fit time, never silently coerced.
- `a ∈ {-1, 1}` for symmetric models.
- `Ks(xi, xj)` negates `xj`, so kernel closures must accept negative inputs even when
  training data is strictly positive.
- `YΓ` is diagonal — add to `diag(Omega)` in place, never build an N×N diagonal matrix.
- Box constraints in Models 1 and 2 are per-sample: `|βk| ≤ 100C/yk`.
- Symmetric models build `Ωs = ½(Ω + a·Ω*)` via `sym_kernel_matrix()` and pass `Ωs`
  straight to the solver — no extra `0.5 *` at the call site. Predictions use
  `sym_kernel_vector()`, which already returns `½ Ks(xk, x)`.
- **NEVER use `$` to read a field on a fit object. Use `x[["name"]]`, or
  `"name" %in% names(x)` to test presence.** `$` partial-matches on lists, so a
  field that does not exist on a class can silently return a *different* field.
  The four classes have different shapes, so a read that is correct on one can
  be wrong on another — and the failure is silent, invisible to `R CMD check`
  and to the test suite.

  The case that proves it — `summary()` tested for symmetry with
  `!is.null(object$a)`:

  | class | what `object$a` returns | why |
  |---|---|---|
  | `psvr_rmspe` | the **length-N `alpha` vector** | `a` uniquely prefix-matches `alpha`; the caller expected `NULL` and got a vector |
  | `psvr_mape` | `NULL` | only because `alpha` **and** `alpha_star` make the prefix ambiguous |
  | `psvr_mape_sym` | `1L` / `-1L` | exact match wins |

  Correct by luck in both directions: right answer on two classes for a reason
  that has nothing to do with the field being absent, wrong answer on the third.

  **How big the hazard actually is — derived, not guessed (2026-08-11).**
  Resolving all 21 field names against all four classes under `pmatch()`
  semantics closes this structurally: the **wrong-object surface is exactly one
  cell** — `psvr_rmspe$a → alpha`, the row above. No other field name can
  produce the shape on any class. That derivation is over a finite vocabulary,
  so it covers read sites nobody has looked at.

  This does **not** make the rule optional. Two things it does not cover:
  *abbreviations* (`$conv`, `$fitt`) are outside the 21 names — 274 such
  strings resolve, all to the right field today, so they are a fragility
  hazard; and nothing here constrains code written against a *future* shape.
  Use `[["name"]]` and `%in% names()` regardless — the derivation is why the
  known surface is small, not a licence to skip the idiom.

  > **CORRECTED 2026-08-11.** This block previously ended: *"A later field named
  > `a…` would silently flip `psvr_mape` too."* **That is false.** An
  > `a`-prefixed field *preserves* the ambiguity and cannot resolve it —
  > `+ active_set`, `+ accel`, `+ a_scale`, `+ aux` all leave `$a` at `NULL`,
  > and `+ a` exactly is an exact match, which is correct behaviour. All five
  > tested. The **only** change that flips `psvr_mape` is **removing
  > `alpha_star`** (or `alpha`), leaving one `a`-prefixed field — and that is a
  > breaking change against a documented `@return`, so it cannot happen quietly.
  > The true fragility scope is far narrower than the sentence claimed.

## Conventions

- **Style**: tidyverse / base R — `snake_case`, `<-` assignment.
- **Docs**: roxygen2 with `@param`, `@return`, `@examples`.
- **Target validation**: every fitter calls `stopifnot(all(y > 0))`.
- **License**: MIT.

## Build and environment

- **R 4.5.3** at `C:\Program Files\R\R-4.5.3\bin\Rscript.exe`.
- From PowerShell: `& "C:\Program Files\R\R-4.5.3\bin\Rscript.exe" script.R`.
  Use forward slashes or single-quoted strings to avoid `\U` Unicode escape errors.
- Load in a session with `devtools::load_all()` from the repo root.
- **osqp 1.0.0** uses `solve_osqp(P, q, A, l, u, pars)` directly — the old R6
  `osqp(...)$solve()` API is gone.
- **`README.md` is regenerated with `rmarkdown::render("README.Rmd")`, NEVER
  `knitr::knit()`.** `README.Rmd` declares `output: github_document`, which is a
  pandoc writer: `render()` applies it, `knit()` skips pandoc entirely. A raw knit
  leaks the YAML front matter into `README.md`, re-wraps prose to ~85 columns
  instead of ~72, strips the alignment padding from tables, and turns ` ``` r `
  fences into ` ```r `. Nothing guards this — the pre-commit hook checks that
  `README.md` is not older than `README.Rmd` and that both are staged, not which
  renderer produced it. After rendering, diff every printed value: the Quick-start
  numbers must reproduce byte-for-byte unless a fitter or solver changed.
  (~~`R/psvr-main.R`~~ — that file was deleted in the 0.0.2.9012 `psvr()` split;
  the numbers now come from `psvr_mape()` / `psvr_rmspe()` and the `.fit_*`
  internals.)
- **`FAIL 0` is NOT a green suite.** The gate is **`FAIL 0` AND `ERROR 0` AND
  `PASS == predicted` AND `SKIP == expected`** — all four, every time. Predict the
  PASS count *before* running and reconcile any difference; do not read the number
  off the run and call it the baseline. testthat's summary line hides two distinct
  failure modes, both of which fired during the stage-5 split:
  - **An errored block counts as `0 passed / 0 failed`.** A checkpoint reported
    `FAIL 0` while **six blocks errored** — the suite looked clean and six blocks
    had not run. Only the PASS count moving against its prediction exposed it.
  - **`skip_on_cran()` silently disables everything under `NOT_CRAN=false`.** The
    same file reported `18 PASS / 0 FAIL / 0 SKIP` under `NOT_CRAN=true` and
    `0 PASS / 0 FAIL / 6 SKIP` under `false`. "No failures" was true of both. CI
    pins `NOT_CRAN: 'false'`, so a gate that does not state its SKIP count can
    certify a run in which nothing executed.
- **Stage by explicit path. NEVER `git add -A` / `git add .`.** Name every file.
  A blanket add sweeps in build artefacts that the working tree happens to be
  carrying, and they reach the tarball. Measured: `rmarkdown::render("README.Rmd")`
  writes a `README.html` preview next to `README.md`, `git add -A` staged it, and
  `R CMD check --as-cran` returned a NOTE — *"Non-standard file/directory found at
  top level: 'README.html'"* — that the package did not have before. `README.Rmd`
  now sets `html_preview: false` and both ignore files list `README.html`, but the
  general rule is the load-bearing one: the next artefact will have a different
  name. Corollary: `R CMD check` counts are only meaningful against a baseline
  **measured on the same harness immediately beforehand**, never against a figure
  inherited from an earlier commit body.
- **Build invariant — do not change**: `src/Makevars` and `Makevars.win` set
  `PKG_LIBS = $(BLAS_LIBS) $(FLIBS)`. `core_smo_solve.cpp` calls `F77_CALL(dgemv)`,
  so removing `$(BLAS_LIBS)` breaks linking. Do not add `$(LAPACK_LIBS)`; if ever
  needed the order must be `$(LAPACK_LIBS) $(BLAS_LIBS) $(FLIBS)` per
  *Writing R Extensions* §1.2.3.

## Known limitation

The built-in SMO solver reaches `max_iter` without converging on some MAPE
fits: it warns, `converged` on the returned fit is `FALSE`, and `iterations`
hits the cap. Several tests document this with "did not converge" warnings.
Root cause is **not established**. Prefer `solver = "osqp"` when a fit does not
converge and accuracy matters.

**CORRECTED 2026-08-26** — the kernel attribution ("linear and polynomial;
RBF is unaffected") was falsified, and the field is `converged`, not
`solver_meta$converged`; see `PSVR_STATUS.md` §2.23. **Do not restate this
limitation as a longer list of kernels** ("linear, polynomial, and RBF at
high p") — the kernel is not the governing variable, so extending the
taxonomy repeats the original error at a larger radius.

---

<!-- Local-only detail: implementation history, solver internals, and research
     notes. The file below is gitignored, so this import is a no-op for anyone
     who clones the repo. -->

@.claude/CLAUDE.md
