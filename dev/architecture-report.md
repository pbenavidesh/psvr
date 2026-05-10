# psvr — Architecture Reconnaissance (pre-F1 refactor)

Read-only audit of the package as of commit `1bd4dea` (branch `main`,
`v0.0.2`). The intent is to map what exists *before* unifying the four
model functions behind a single `psvr()` API and integrating the six
efficiency theorems from arXiv:2605.01446 v3.

---

## 1. File inventory

All `.R` files in `R/`. Total: **9 files / 2 527 lines**.

| Role                | File                  | LOC | One-line purpose                                                                                                                  |
|---------------------|-----------------------|----:|-----------------------------------------------------------------------------------------------------------------------------------|
| Solver — ε-SVR      | `mape_svr.R`          | 251 | Model 1 fit / predict / print / coef. SMO (default) or `osqp` backend, bias recovery, SV pruning.                                 |
| Solver — ε-SVR sym. | `mape_sym_svr.R`      | 261 | Model 2. Same backbone as Model 1 but builds `Ks = Ω + a·Ω*` and absorbs the `½` representer factor into `Omega = ½·Ks` for SMO.  |
| Solver — LS-SVR     | `rmspe_lssvr.R`       | 240 | Model 3. Solves the bordered `(N+1)×(N+1)` linear system; optional Remark-17 preconditioner `P = diag(1/y)`.                      |
| Solver — LS-SVR sym.| `rmspe_sym_lssvr.R`   | 229 | Model 4. Same as Model 3 with `Ωs = ½(Ω + a·Ω*)` substituted for `Ω`.                                                             |
| SMO core            | `smo_solve.R`         | 229 | Internal `.smo_solve()`: libsvm-style WSS1+WSS3 working-set selection, shrinking, bias recovery. Shared by Models 1 & 2.          |
| Kernel              | `kernel.R`            | 131 | `make_kernel()` closure (RBF / linear / polynomial), `kernel_matrix()`, `sym_kernel_matrix()` (returns `Ωs` *with* ½), `sym_kernel_vector()`. |
| parsnip integration | `parsnip.R`           | 853 | 12 fit-wrapper functions, 12 `model_spec` constructors, 12 `update.*` methods, `make_psvr_engines()` registration helper.         |
| dials params        | `params.R`            | 327 | Custom dials params (`cost_psvr`, `margin_percentage`, `rbf_sigma_psvr`, `sym_type_param`); workflow-set helpers.                 |
| Package doc         | `psvr-package.R`      |   6 | `_PACKAGE` stub; no real code.                                                                                                    |

Test files live in `tests/testthat/` (total: 9 files / 1 273 lines, see §6).

---

## 2. The four model functions

### 2.1 `mape_svr()` — Model 1

- **File:** `R/mape_svr.R:51–172`
- **Signature:**
  ```r
  mape_svr(X, y, kernel, C, eps,
           solver = c("smo", "osqp"), tol = 1e-5)
  ```
- **Pseudocode:**
  ```
  validate y > 0, C > 0, eps >= 0
  warn if N > 2000
  Omega <- kernel_matrix(kernel, X);  diag(Omega) += 1e-6
  if solver == "smo":
      sol <- .smo_solve(Omega, y, C, eps)
      beta <- sol$alpha - sol$alpha_star;  b <- sol$b
  else (osqp):
      build P = [Ω,-Ω;-Ω,Ω] (sparse, upper triu), q = [y(ε/100-1); y(1+ε/100)]
      A = [eq-row [1,-1]; box I_{2N}],  l/u with per-sample ub = 100C/y
      res <- osqp::solve_osqp(P, q, A, l, u)
      beta <- res$x[1:N] - res$x[N+1:2N]
      recover b from free SVs (else KKT sandwich on saturated SVs)
  prune SVs with |beta| <= tol
  return structure(list(beta, b, X_sv, y_sv, kernel, C, eps,
                        n_train, p_train), class = "psvr_mape")
  ```
- **Helpers called:** `kernel_matrix()`, `.smo_solve()`, `osqp::solve_osqp()`, `Matrix::*`. Print path uses `.kernel_desc()` (defined inline in this file).

### 2.2 `mape_sym_svr()` — Model 2

- **File:** `R/mape_sym_svr.R:60–187`
- **Signature:**
  ```r
  mape_sym_svr(X, y, kernel, C, eps, a = 1,
               solver = c("smo", "osqp"), tol = 1e-5)
  ```
- **Pseudocode:** identical to Model 1 except for the kernel block:
  ```
  Omega     <- kernel_matrix(kernel, X, X)
  Omega_neg <- kernel_matrix(kernel, X, -X)
  Ks        <- Omega + a * Omega_neg          # NO ½ factor here
  diag(Ks) += 1e-6
  if solver == "smo": sol <- .smo_solve(0.5 * Ks, ...)   # ½ absorbed into "Omega"
  else (osqp):       P = ½ [Ks,-Ks;-Ks,Ks]               # so ½·u'Pu = ¼ β'Ksβ
                     bias recovery uses ½(Ks β)
  ```
  Validates `a ∈ {-1, 1}`.
- **Helpers called:** `kernel_matrix()`, `.smo_solve()`, `osqp::solve_osqp()`. Note: it does **not** use `sym_kernel_matrix()` (which has a baked-in ½ — see §4).

### 2.3 `rmspe_lssvr()` — Model 3

- **File:** `R/rmspe_lssvr.R:78–143`
- **Signature:**
  ```r
  rmspe_lssvr(X, y, kernel, gamma, precondition = "auto")
  ```
- **Pseudocode:**
  ```
  validate y > 0, gamma > 0
  use_precond <- .resolve_precondition(precondition, y)   # rho = max(y)/min(y)
  Omega <- kernel_matrix(kernel, X);  diag(Omega) += 1e-6
  if use_precond:
      P = 1/y;  Omega <- (P %o% P) * Omega;  diag(Omega) += 1/gamma
      border <- P;  rhs_y <- P*y       # = 1
  else:
      diag(Omega) += y^2 / gamma       # YΓ added in place
      border <- rep(1, N);  rhs_y <- y
  build (N+1)×(N+1) bordered matrix [[0, border'],[border, Omega]]
  sol <- solve(A, c(0, rhs_y))
  alpha <- sol[-1];   if (use_precond) alpha <- alpha / y
  return structure(list(alpha, b, X_train, kernel, gamma,
                        n_train, p_train, precondition_applied),
                   class = "psvr_rmspe")
  ```
- **Helpers called:** `kernel_matrix()`, `.resolve_precondition()` (defined in this file, lines 217–240; **shared** with Model 4 via package-internal scope).

### 2.4 `rmspe_sym_lssvr()` — Model 4

- **File:** `R/rmspe_sym_lssvr.R:88–155`
- **Signature:**
  ```r
  rmspe_sym_lssvr(X, y, kernel, gamma, a = 1, precondition = "auto")
  ```
- **Pseudocode:** identical to Model 3 except:
  ```
  Omega_s <- sym_kernel_matrix(kernel, X, a)   # = ½(Ω + a·Ω*)
  ... rest unchanged, with Omega_s in the inner block ...
  ```
  Adds `a ∈ {-1, 1}` validation; result class is `"psvr_rmspe_sym"` and carries `a` in the slot list.
- **Helpers called:** `sym_kernel_matrix()`, `.resolve_precondition()`.

### 2.5 Code duplication across the four

Pure copy/paste fragments (essentially line-for-line):

- **`y > 0` validator** — 8 lines, repeated 4×.
- **`N > 2000` warning** — 6 lines, repeated 4×.
- **`predict.psvr_*()` body** — column-count check + per-row `kernel_matrix()` loop, repeated 4× (Models 3/4 use `X_train`, Models 1/2 use `X_sv`; otherwise identical).
- **`coef.psvr_*()` body** — same shape (3-element list) repeated 4×.
- **`print.psvr_*()` body** — same `sprintf` skeleton, 4 variants.
- **`.kernel_desc()` helper** — defined once in `mape_svr.R:243–251` and reused by the other three print methods (which load fine because it's package-internal). This is the only obvious *anti-duplication* in the current codebase.

Models 1 & 2 differ from Models 3 & 4 in solver path (QP vs. linear system) but share scaffolding (`y > 0`, jitter, S3 class returned, predict shape). Models 1 vs. 2 and 3 vs. 4 differ only by kernel-matrix construction (`Ω` vs. `Ks`/`Ωs`) and one extra arg (`a`). Net duplication, eyeballed: **~120–150 lines** that could be lifted into shared helpers, plus another ~60 in print/coef methods.

---

## 3. SMO loop architecture (ε-SVR path)

- **Main loop:** `R/smo_solve.R:32–229`, single function `.smo_solve(Omega, y, C, eps, tol = 1e-3, max_iter = 100000L, n_check = NULL, n_freeze = 5L)`. Default `n_check = min(N, 1000L)`.
- **Working-set selection:**
  - **First variable (i):** WSS1 = `argmax τ` over `I_up` (`smo_solve.R:75–85`). Pool spans both `α` and `α*` via separate index lists.
  - **Second variable (j):** WSS3 — among `I_down` candidates with `τ_t < τ_i`, picks `argmax (τ_i − τ_t)² / a_pt`, where `a_pt = Ω[p,p] − 2 Ω[p,t] + Ω[t,t]` is the 1-D curvature (Fan, Chen & Lin 2005, eq. 14). Falls back to WSS1 partner if no candidate satisfies the strict gap (`smo_solve.R:115–129`).
- **Two-variable update:** closed-form. Step `δ = min(Δ_pq / η, δ_max)` if curvature `η > 0`, else `δ_max`; `δ_max = min(R_i, R_j)` honours the per-sample box `[0, 100C/y_k]` (`smo_solve.R:134–141`). Four update branches handle the four `(i_is_alpha, j_is_alpha)` parity combinations (`smo_solve.R:143–147`). After the update each touched variable is clipped back to `[0, 100C/y]` to absorb up to 1 ulp of drift.
- **Shrinking:** implemented (`smo_solve.R:160–182`). Every `n_check` iterations, mark variables that have stayed at a boundary with a "wrong-side" τ for `n_freeze` consecutive checks as inactive. Active sets `active_alpha`, `active_astar` are masks; the τ vectors are updated only over the active indices each step. Reactivation happens when the active-set gap drops below `tol`: τ is rebuilt from scratch via `Omega %*% beta`, masks reset, and the loop continues (`smo_solve.R:98–110`).
- **Convergence check:** `Δ = τ_i − min(τ over I_down) ≤ tol_eff`, where `tol_eff = tol · mean(y)` so the threshold scales with target magnitude (`smo_solve.R:44, 96–113`). When the active-set gap meets this and any variables are still shrunk, the rebuild branch reactivates them; only when a *full* set passes does `converged <- TRUE`.
- **Bias recovery:** mean of τ over free SVs; falls back to the `(min I_down + max I_up)/2` sandwich if no free SV exists (`smo_solve.R:196–220`). Mirrors the osqp-path bias logic in `mape_svr.R`/`mape_sym_svr.R`.

---

## 4. Kernel infrastructure

**Critical question — does psvr have a kernel column cache?** **No.** The kernel matrix is materialized once.

- `kernel_matrix(K, X1, X2 = X1)` (`kernel.R:80–90`) is a plain double-`for` loop over `(i, j)` returning a dense numeric matrix.
- All four model fitters call `kernel_matrix(kernel, X)` *once* at fit time and keep the full N×N (or its symmetric variant) in memory; the SMO loop reads `Omega[p, :]`, `Omega[:, q]`, and `diag(Omega)` directly out of that dense matrix (`smo_solve.R:46, 123, 135, 156`).
- Predictions are **also un-cached**: `predict.psvr_*()` recomputes a fresh kernel row per query point inside an R-level `for` loop (`mape_svr.R:198–203`, `mape_sym_svr.R:217–223`, `rmspe_lssvr.R:169–173`, `rmspe_sym_lssvr.R:184–188`). Each call does `M` calls to `kernel_matrix()` of shape `n_sv × 1`, i.e., `M × n_sv` calls into the kernel closure.
- **Eviction policy:** N/A — no cache exists.
- **API the SMO loop uses:** raw matrix indexing on the dense `Omega` argument. Adding a column cache would require either (a) replacing `Omega` with an accessor object that responds to `[, q]`, `diag()`, and `[p, q]`, or (b) routing the SMO loop through dedicated getters.

A `2000 × 2000` warning is emitted in every fitter (`mape_svr.R:69`, `mape_sym_svr.R:79`, `rmspe_lssvr.R:96`, `rmspe_sym_lssvr.R:105`) — explicit acknowledgement that the full-materialization design is the current scalability ceiling.

### Symmetric kernels (Models 2 & 4)

- **`Ωs = ½(Ω + a·Ω*)` location:** *Two different conventions exist*, which is a known landmine.
  - **Model 4 (`rmspe_sym_lssvr.R:113`)** uses the helper `sym_kernel_matrix(K, X, a)` (`kernel.R:104–108`), which returns `Ωs` *with* the `½` factor baked in.
  - **Model 2 (`mape_sym_svr.R:88–91`)** builds `Ks = Ω + a·Ω*` *inline*, **without** the `½`. The SMO path then passes `0.5 * Ks` to `.smo_solve()` (so the solver's `Omega` argument is `Ωs = ½·Ks` again), while the osqp path absorbs the `½` into `P = ½ [Ks,-Ks;-Ks,Ks]` so that osqp's internal `½ uᵀPu` evaluates to `¼ βᵀKsβ`.
- **`Ω*` storage:** computed on the fly via `kernel_matrix(K, X, -X)` and *immediately fused* into the symmetric matrix. The unsymmetrized `Ω*` is never stored on the model object — only `Ks` (Model 2) or `Ωs` (Model 4), and even those are only kept inside the fit closure; the returned object retains just `kernel`, `X_sv`/`X_train`, and `a`. Predictions reconstruct `½ Ks(x_k, x_new)` per query point via `sym_kernel_vector()` (`kernel.R:124–131`), again without caching.

This split convention (`Ks` no-½ in Model 2, `Ωs` with-½ in Model 4) is documented in `CLAUDE.md` and in `mape_sym_svr.R:14–18`, but it is exactly the kind of asymmetry a unified `psvr()` API would have to either preserve or normalise.

---

## 5. Parsnip integration

All in `R/parsnip.R` (853 lines). Three layers stacked top-down.

### 5.1 Fit wrappers (`parsnip.R:47–155`)

Twelve exported `psvr_<loss>_<kernel>_fit()` functions. Each takes a parsnip `(x, y, ...)` matrix interface, builds the kernel via `make_kernel()`, and forwards to the corresponding model function. Examples:

```r
psvr_mape_rbf_fit  <- function(x, y, C, eps, rbf_sigma = 1, tol = 1e-5)
                       mape_svr(X = x, y = y,
                                kernel = make_kernel("rbf", sigma = rbf_sigma),
                                C = C, eps = eps, tol = tol)

psvr_rmspe_sym_rbf_fit <- function(x, y, gamma, rbf_sigma = 1,
                                    sym_type = "even",
                                    precondition = "auto") { ... }
```

`sym_type ∈ {"even","odd"}` is translated to `a ∈ {1L,-1L}` inside the wrapper.

### 5.2 Spec constructors and `update()` methods (`parsnip.R:158–717`)

- **12 `psvr_<loss>_<kernel>()` model-spec constructors**, each calling `parsnip::new_model_spec(<class>, args, eng_args = NULL, ...)`.
- **12 `update.<class>()` methods** delegating to a private `psvr_update_spec()` helper that uses only public parsnip API (no `:::`).

### 5.3 Engine registration (`parsnip.R:720–844`)

- A `.reg_psvr(model_name, fit_fun, arg_defs, defaults)` helper does the six standard parsnip calls (`set_new_model`, `set_model_mode`, `set_model_engine`, `set_dependency`, `set_fit`, `set_pred`) plus per-arg `set_model_arg`.
- Reusable arg definitions (`.A_COST_C`, `.A_COST_GAMMA`, `.A_MARGIN`, `.A_SIGMA`, `.A_DEGREE`, `.A_SCALE`, `.A_SYM_TYPE`) are list constants on lines 793–799.
- `make_psvr_engines()` calls `.reg_psvr` 12 times. Triggered from `.onLoad()` (`parsnip.R:849–853`); guarded against double-registration.

### 5.4 Engine vs. tunable args

| Argument        | Class       | Surfaced in spec? | dials param                                  |
|-----------------|-------------|-------------------|----------------------------------------------|
| `cost` (→ C/Γ)  | tunable     | yes               | `cost_psvr` (range `[-2, 10]` log2, custom)  |
| `svm_margin`    | tunable     | yes (MAPE only)   | `margin_percentage` (range `[1, 20]` linear) |
| `rbf_sigma`     | tunable     | yes (RBF only)    | `rbf_sigma_psvr` (range `[-3, 1]` log10)     |
| `degree`        | tunable     | yes (poly only)   | `dials::degree`                              |
| `scale_factor`  | tunable     | yes (poly only)   | `dials::scale_factor`                        |
| `sym_type`      | tunable     | yes (sym only)    | `sym_type_param` (qual: `even`/`odd`)        |
| `tol`           | engine arg  | via `set_engine`  | —                                            |
| `precondition`  | engine arg  | via `set_engine`  | — (default `"auto"` registered via `defaults = list(precondition = "auto")`) |
| `a` (poly/lin)  | engine arg  | via `set_engine`  | — (defaults to `1L` for sym poly/linear)     |

### 5.5 Shared prediction infrastructure

- All 12 specs use the same `set_pred()` block (`parsnip.R:760–771`) that calls `predict(object$fit, newdata = new_data)` — i.e. it relies entirely on the four S3 `predict.psvr_*()` methods. There is **no** parsnip-side post-processing.
- Implication for refactor: a unified `psvr()` would need either a single `predict.psvr()` method (dispatching internally on a sub-type field) or to keep the four S3 classes and have `psvr()` return one of them.

---

## 6. Tests inventory

`tests/testthat/` has **9 files / 1 273 lines / 115 `test_that()` blocks** (counted via `grep -c '^test_that'`). `tests/testthat/_snaps/` is empty — no testthat snapshots are in use.

| File                          |  blocks | Coverage area                                                                       |
|-------------------------------|--------:|-------------------------------------------------------------------------------------|
| `test-kernel.R`               |       8 | `make_kernel()`, `kernel_matrix()`, `sym_kernel_matrix()`. Asserts Assumption 3 holds for RBF, fails for linear. |
| `test-mape-svr.R`             |      13 | Model 1: shape, S3 fields, validation errors, box constraints, `eps=0` tighter than `eps=10`, print/coef. |
| `test-mape-sym-svr.R`         |      14 | Model 2: same coverage as Model 1, plus `a ∈ {-1,1}` validation and "Model 2 ≠ Model 1 predictions". |
| `test-rmspe-lssvr.R`          |      11 | Model 3: shape, validation, KKT identity `f(x)+e=y`, `Σ α = 0`, print/coef.        |
| `test-rmspe-sym-lssvr.R`      |      12 | Model 4: same as Model 3 plus `a` validation and `a = -1` smoke.                   |
| `test-smo-solve.R`            |       6 | SMO ↔ osqp parity (`max_diff < 0.01·mean(y)`), parsnip default smoke, no-free-SV bias fallback, max-iter warning. |
| `test-preconditioner.R`       |      14 | `precondition ∈ {"auto","always","never",numeric}`; "always" ≈ "never" to machine eps at moderate ρ; `auto` activates at ρ > 10. |
| `test-parsnip.R`              |      21 | All 12 specs: `fit_xy` + `predict` smoke; `precondition` engine-arg forwarding for the 4 RMSPE specs (rbf/poly/linear/sym_rbf). |
| `test-params.R`               |      16 | `cost_psvr`, `margin_percentage`, `rbf_sigma_psvr`, `sigma_heuristic`, `cost_psvr_ls_data`, `psvr_option_add*` helpers. |

### Golden-output ("bit-identical") tests — **the critical question**

There are **no** stored numerical fixtures (no `_snaps/`, no `expect_equal_to_reference`, no hard-coded coefficient values). The numeric assertions used today are:

1. **KKT identities** — `f(x_k) + (y_k²/Γ)·α_k = y_k` (`tolerance = 1e-6`), `Σ α = 0` (`tolerance = 1e-10`). Pass for any solver that reaches the optimum to that precision.
2. **Cross-solver parity** — SMO vs. osqp in `test-smo-solve.R` uses `max(|p_smo − p_osqp|) < 0.01 · mean(y)` (i.e., 1 % of mean target), not exact equality.
3. **Preconditioner equivalence** — `precondition="always"` vs. `"never"` checked to machine epsilon at moderate ρ and a looser tolerance at extreme ρ (`test-preconditioner.R`).
4. **Monotonicity / direction** — `mape_svr(eps=0)` MAPE < `mape_svr(eps=10)` MAPE, etc.
5. **Shape / class / validation** — purely structural.

**Implication for the refactor.** There is currently **no test that pins exact predictions or coefficients**. A bit-identical refactor acceptance criterion would have to be *added* before the F1 work starts — for example by snapshotting `predict()` outputs of all four models on a fixed seed/dataset/hyperparameters into `_snaps/`. Today's suite would silently accept ~1 % drift on Models 1/2 and ~1e-6 drift on Models 3/4.

---

## 7. Refactor opportunity assessment

### (a) Genuinely duplicated code (consolidation wins)

- **`y > 0` and parameter validation** (4×, ~30 lines total) — trivially extracted into one `.validate_psvr_inputs(X, y, C = NULL, gamma = NULL, eps = NULL, a = NULL)` helper.
- **`N > 2000` warning** (4×, ~24 lines) — same idea.
- **`predict.psvr_*()` per-row loop** — Models 1/2 differ from 3/4 only in field name (`X_sv` vs. `X_train`) and the symmetric branch in the kernel call. A single helper `.psvr_predict(object, newdata, sv_field, kernel_eval)` would cover all four.
- **`coef.psvr_*()`** — identical 3-element list with one field rename.
- **`print.psvr_*()`** — `sprintf` template differs only in header/values.
- **Bordered linear-system construction** — Models 3 and 4 have copy/paste blocks (`rmspe_lssvr.R:117–125` ↔ `rmspe_sym_lssvr.R:128–134`) that differ only in `Omega` vs. `Omega_s`.
- **osqp-path bias recovery in Models 1/2** — `mape_svr.R:127–152` and `mape_sym_svr.R:140–166` differ only in whether `Kbeta` or `½ Ksbeta` is used. The `free_up`/`free_lo`/sandwich logic is line-identical.

### (b) Similar-but-different (looks duplicated, has meaningful divergences)

- **Symmetric kernel construction.** Model 2 builds `Ks = Ω + a·Ω*` *no-½* and absorbs ½ into the QP / SMO inputs; Model 4 calls `sym_kernel_matrix()` which returns `Ωs = ½(Ω + a·Ω*)`. The divergence is *intentional* and load-bearing for the `¼ βᵀKsβ` coefficient in Theorem 2 vs. the plain `Ωs + YΓ` block in Theorem 4. A naïve unification (e.g. always using `sym_kernel_matrix()`) silently breaks Model 2.
- **Diagonal regularisation.** ε-SVR adds `1e-6` Tikhonov jitter. LS-SVR adds `1e-6` jitter *and* either `y²/Γ` (no precondition) or `1/Γ` (with precondition). Looks similar, semantically distinct.
- **Bias recovery.** SMO-path bias uses `mean(τ_free)` over a single τ vector; osqp-path uses `mean(y(1±ε/100) − Kβ)` over two free sets. Both end up at the same `b` algebraically (since `τ = y(1±ε/100) − Kβ` at free SVs), but they touch different intermediate variables.
- **`precondition` resolution.** `.resolve_precondition()` lives in `rmspe_lssvr.R:217–240`; only LS-SVR models use it. Carrying it into a unified `psvr()` is fine, but it must remain a no-op for ε-SVR.

### (c) Hidden dependencies that may break a unified-API refactor

- **S3 class names are part of the parsnip contract.** `set_pred()` (`parsnip.R:760–771`) calls `predict(object$fit, ...)`, which dispatches on `class(object$fit)`. The four S3 classes (`psvr_mape`, `psvr_mape_sym`, `psvr_rmspe`, `psvr_rmspe_sym`) plus their 4 `predict.*` methods are load-bearing. Replacing them with a single `psvr` class means writing one `predict.psvr()` that handles all four sub-types.
- **Field-name asymmetry.** `coef.psvr_mape*()` returns `list(alpha, b, X_sv)` (re-keying `beta` → `alpha`); `coef.psvr_rmspe*()` returns `list(alpha, b, X_sv)` (re-keying `X_train` → `X_sv`). Both `coef()` outputs are `X_sv`, but the model objects themselves expose `X_sv` (ε-SVR) vs. `X_train` (LS-SVR). A user (or downstream package) reading `fit$X_sv` works for ε-SVR only; `fit$X_train` works for LS-SVR only. The `test-mape-svr.R:15` and `test-rmspe-lssvr.R:11` `expect_named()` calls hard-code these names — those tests will fail under any field-rename.
- **`expect_named()` exact lists.** Each of the four "shape and class" tests uses `expect_named(fit, c(...))` with the *exact* full slot list. Adding any field (e.g. a `model_type` discriminator for the unified API) will break all four tests.
- **`fit_obj$fit$precondition_applied`.** `test-parsnip.R:111–169` reads this slot directly. The slot must survive the refactor unchanged for LS-SVR fits.
- **`psvr:::.smo_solve` is referenced in tests.** `test-smo-solve.R:118` uses the `:::` triple-colon to call the internal solver directly. Renaming it (e.g. to `.psvr_smo`) will break that test.
- **`make_psvr_engines()` registers by model-spec class string** (`"psvr_mape_rbf_model"` etc.). Any change to spec-class names propagates to `parsnip.R`, all 12 `update.*` methods, and the persistent parsnip env between `devtools::load_all()` calls (the `.onLoad` hook has a guard for this).

### (d) Trickiest part of unifying the four functions

In rough order of difficulty:

1. **Reconciling the two symmetric-kernel conventions** (no-½ `Ks` for the QP/SMO path vs. with-½ `Ωs` for the linear-system path). Either standardise on one (and rewrite the other path to compensate) or keep both and select inside the unified entry point — but the choice has to be made *consciously* so the Theorem-2 ¼ coefficient is preserved.
2. **Argument-shape unification.** ε-SVR takes `(C, eps, [a], solver, tol)`; LS-SVR takes `(gamma, [a], precondition)`. A `psvr(loss = c("mape","mape_sym","rmspe","rmspe_sym"), ...)` signature has to either accept a superset (and ignore irrelevant ones with a warning) or expect the user to pass a single named list (`hp = list(C = 10, eps = 5)`) — both have parsnip implications because each tunable arg is currently surfaced via `set_model_arg` on a per-model-class basis.
3. **Keeping the parsnip surface stable.** The 12 specs are public API. Even if `psvr()` becomes the new internal entry point, the 12 fit wrappers and 12 spec constructors should keep their names and signatures, otherwise downstream tuning code (and the website / vignettes that demonstrate them) breaks. This makes the refactor a *non-breaking internal consolidation* rather than an API change.
4. **Single `predict.psvr()` design.** Needs a discriminator field on the fit object (e.g. `loss`, `solver_family`, `is_symmetric`) that the dispatch reads. Today's `class()` is the discriminator; replacing it with a field requires updating both the predict logic and any test that asserts on `class()` or named-slot contents.
5. **Backwards-compatibility on the model object.** Either keep `beta`/`alpha`/`X_sv`/`X_train` as the original four S3 classes had them (and let `psvr()` return one of those four classes), or normalise to a single shape (and accept that `expect_named()` tests will need updating). The cheapest path is the former — `psvr()` is just a router that returns whichever of the four existing classes matches the requested loss/symmetry combo, keeping every existing test green.

---

## 8. Open questions for the package author

1. **Acceptance criterion for "bit-identical".** What tolerance qualifies as "identical" in the refactor? `0` (literal `identical()`), machine-eps, or the `1e-6`/`1 % mean(y)` tolerances already used? The current suite cannot detect drift below those bounds.
2. **Should `psvr()` return one of the four existing S3 classes, or a single new `psvr` class?** This single decision drives ~80 % of the test churn and the parsnip integration changes.
3. **Standardise on `Ks` (no-½) or `Ωs` (with-½)?** Both conventions co-exist today. A single internal representation would simplify the upcoming Algorithm 2 / efficiency-theorem work.
4. **Do you want a kernel column cache as part of F1, or strictly later?** Today there is none. Algorithm 2 (adaptive spectral regularisation) and several arXiv:2605.01446 efficiency theorems normally assume one. Adding a cache while consolidating might be cleaner than two separate refactors but enlarges F1's scope.
5. **`X_sv` vs. `X_train` slot.** ε-SVR prunes to support vectors; LS-SVR keeps all N rows (every α_k is a "support vector" in LS-SVR). Should the unified object always expose `X_sv` (with `n_sv == n_train` for LS-SVR), always expose `X_train`, or expose both?
6. **`precondition` for ε-SVR?** Remark 17 currently applies only to LS-SVR. Is it expected to extend to ε-SVR in F1+, or stay LS-SVR-only? Affects whether the unified API surfaces `precondition` for all losses or selectively.
7. **`a` parameter for non-symmetric models.** Today only the two `_sym_` models accept `a`. In a unified `psvr()`, is `a = NULL` the contract for non-symmetric ("ignore"), or should `a` move to a `symmetry = c("none","even","odd")` enum?
8. **Is the parsnip surface frozen?** I.e., are the 12 `psvr_<loss>_<kernel>()` constructors and the 12 spec classes considered public API for v1.0, or can F1 collapse them (e.g. into 4 specs with kernel as a tunable)? The tests in `test-parsnip.R` (lines 26–92) currently pin all 12.
