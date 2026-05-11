# F7 — Block-k=4 SMO with descent-guaranteed decoupling

**Status:** Archived implementation notes — current state lives in `CLAUDE.md`.

## Block-k=4 SMO with descent-guaranteed decoupling (post-F7)

F7 implements Theorem 7 of arXiv:2605.01446 v3 ("strictly novel"
theorem of the smo-v3 paper). After WSS1 + WSS3 pick pair 1
`(i_1, j_1)`, the solver tries to find a second sample-disjoint pair
`(i_2, j_2)` and apply a 2-D joint update governed by a descent-
guaranteed decoupling criterion. When the test fails the iteration
falls back to the standard 1-D pair-1 update (k=2 path).

### Δβ invariant (drives the corrected xi formula)

For all 4 cases of the SMO 1-D update (alpha-alpha, alpha-alpha*,
alpha*-alpha, alpha*-alpha*), the change in `β = α − α*` is:

```
Δβ_{p_i} = +δ,  Δβ_{p_j} = −δ,  all other Δβ_k = 0
```

The slot (α vs α*) is absorbed by the equality constraint
`sum(α − α*) = 0` and the four case branches; it does NOT appear in
the β update. This is what makes the corrected sign-free xi formula
valid and what makes the tau update use the same `diff_pq` vector
for both `tau_alpha` and `tau_alphastar`.

### xi formula (corrected, sign-free)

```
xi / (δ_1 δ_2) = Ω(p_1, p_2) − Ω(p_1, q_2) − Ω(q_1, p_2) + Ω(q_1, q_2)
```

No sign factors `s_i s_j`. They cancel under the α/α* dual. The
formula is computable from the already-fetched K_p (column for
sample p_1) and K_q (column for sample q_1) with **zero extra column
fetches**: `xi = K_p[p_2] - K_p[q_2] - K_q[p_2] + K_q[q_2]`.

The spec given to me used `s_{i1}*s_{i2}*Ω(...) − ...` with explicit
sign factors; that form is the cross-curvature for a *different* dual
structure (e.g., standard C-SVC), not MAPE-SVR's α/α* split.
Implementation uses the simplified form; this is a paper-side
clarification (paper TODO update).

### Descent test (D1)

```
Δ_pq^2 · η_2 + Δ_2^2 · η_1 > 2 |xi · Δ_pq · Δ_2| · (1 + 1e-10)
```

Where `Δ_pq = τ_i − τ_q` is pair 1's WSS3 gap (the value used to
compute `δ_1 = Δ_pq / η_1`), NOT the WSS1 convergence gap
`τ_i − τ_{j_w1}`. The descent geometry of the joint update is
defined by pair 1's actual update direction `(i, q)`, not the MVP
convergence pair `(i_w1, j_w1)`.

This is a necessary-and-sufficient criterion (in exact arithmetic)
for the joint update to outperform the k=2 fallback. No tuning
threshold required.

### Pair-2 selection (D2)

```
i_2 = argmax τ over I_up \ {i_1}                       (sample-disjoint from i_1)
j_2 = argmax score_j_aug over I_down filtered          (sample-disjoint from p, q, p2)
  where score_j_aug = gain_j × (1 − alpha_couple × coupling_j)
        gain_j      = (τ_{i_2} − τ_j)^2 / max(η_j, 1e-12)
        coupling_j  = |Ω(p, j)| / sqrt(Ω(p,p) · Ω(j,j))  (1 if denom ≤ 0)
```

`alpha_couple` defaults to `0.5`; exposed as an internal parameter
on `psvr()` for empirical tuning. The coupling proxy uses the K_p
column already on hand — zero extra fetches for scoring.

Sample-level disjointness (not slot-level) ensures `Δβ_1` and `Δβ_2`
have disjoint support, keeping the xi formula clean.

### Telemetry

`psvr_fit$solver_meta` extends with:
- `joint_updates` — number of iterations where the joint update was applied.
- `k2_fallbacks` — iterations that fell back to k=2 after entering the
  block-k=4 path.
- `decoupling_rate` — `joint_updates / (joint_updates + k2_fallbacks)`.
- `early_phase_decoupling_rate` — rate over the first ¼ of iterations
  (floor 50).
- `late_phase_decoupling_rate` — rate over the last ¼ (floor 50).

The early/late split tests the paper's "clustered active set in the
late phase" hypothesis. Empirical observation: late ≥ early by 5–15
percentage points across all regimes — the asymmetry is real but
modest, not the dramatic clustering the paper might predict.

### Per-iter wall — the central F7-C-full finding

The R implementation imposes a 2× per-iter wall overhead from the
block-k=4 scaffolding (pool subsetting, `which.max` calls, `ifelse`
coupling computation). The C++ port reduces this to 1.40×:

| Regime | F4-R ms/iter | F4-Rcpp ms/iter | F7-R ms/iter | F7-Rcpp ms/iter | F7/F4 R | **F7/F4 Rcpp** |
|--------|:----:|:----:|:----:|:----:|:----:|:----:|
| R1 N=1000 RBF σ=1 (max_iter)| 0.104 | 0.045 | 0.223 | 0.063 | 2.14× | **1.40×** |
| R2 N=1000 RBF σ=1 ρ_y=3.6   | 0.108 | 0.047 | 0.226 | 0.060 | 2.09× | **1.28×** |
| R3 N=300 RBF σ=3            | 0.044 | 0.009 | 0.094 | 0.012 | 2.14× | **1.33×** |
| R4 N=1000 RBF σ=0.3 (converges)| 0.157 | 0.098 | 0.300 | 0.155 | 1.91× | **1.58×** |

Combined with 38–48% iter reduction on converging regimes, this gives
+12.2% (R1) and +17.5% (R4) wall-positive translation. **Paper TODO
#9 (T7 wall regression) is RESOLVED on converging regimes by the
C++ port.**

### Per-regime bench summary (engines × modes)

| Regime | F4-R wall | F4-Rcpp wall | F7-R wall | F7-Rcpp wall | F4 R→Rcpp | F7 R→Rcpp | F7 vs F4 (Rcpp) |
|--------|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| R1 N=1000 ρ_y=44k σ=1 | 10.37 s | 4.49 s | 13.88 s | 3.94 s | 2.31× | 3.52× | **+12.2%** wall |
| R2 N=1000 ρ_y=3.6 σ=1 | 10.76 s | 4.65 s | 22.56 s | 6.00 s | 2.31× | 3.76× | -28.9% (both max_iter) |
| R3 N=300 ρ_y=3702 σ=3 | 4.40 s | 0.94 s | 9.36 s | 1.21 s | 4.69× | 7.73× | -29.2% (both max_iter) |
| R4 N=1000 ρ_y=13901 σ=0.3 | 0.232 s | 0.145 s | 0.232 s | 0.120 s | 1.60× | 1.94× | **+17.5%** wall |

R2/R3 are dominated by the pre-existing SMO non-convergence pathology
(both engines hit `max_iter = 100 000`; paper TODO #5). T7 cannot help
when neither engine converges.

### T5 × T7 stacking — B-suite at N=300, 10-fold CV (ρ_y=2388)

| ID | engine | warm | bk4 | wall | iter_sum | speedup vs B1-r |
|----|--------|------|-----|------|----------|:--:|
| B1-r | r | TRUE | FALSE | 2.200 s | 37 740 | 1.00× (paper baseline) |
| B1-rcpp | rcpp | TRUE | FALSE | 0.591 s | 37 740 | **3.72×** |
| B2-r | r | FALSE | TRUE | 2.697 s | 23 363 | 0.82× |
| B2-rcpp | rcpp | FALSE | TRUE | 0.508 s | 23 363 | **4.33×** |
| B3-r | r | TRUE | TRUE | 2.489 s | 26 586 | 0.88× |
| **B3-rcpp** | rcpp | TRUE | TRUE | **0.515 s** | 26 586 | **4.28×** |

B3-rcpp is the current default (`psvr_cv()` with warm-start +
block-k=4 + Rcpp). T5 × T7 non-multiplicative stacking is confirmed
under both engines: B2 (T7 alone) is marginally faster wall AND has
lower iter_sum than B3 (T7 + warm-start). The non-multiplicativity is
algorithmic, not implementation-specific — paper TODO #10 has bi-
engine evidence.

### Default-collapse

`block_k4_enabled = FALSE` reduces the SMO inner loop to F4 behaviour
bit-identically. The path is preserved through the engine dispatcher
on BOTH `engine = "r"` and `engine = "rcpp"`. The F4 baseline gate
(`_snaps/block-k4.md`) protects this invariant.

### Design decisions (paper-relevant)

- **Sample-level disjointness** between pair 1 and pair 2 (stricter
  than dual-variable level disjointness). User-resolved during
  planning. Simpler and safer; keeps the xi formula clean (no
  diagonal Ω(p, p) terms from sample collisions).
- **`alpha_couple = 0.5` default** — empirically influences but does
  not dominate the decoupling rate (which is 0.93–1.0 across regimes
  regardless of `alpha_couple`).
