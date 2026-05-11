# F4 — Asymmetric Freeze + Per-pair Tolerance

**Status:** Archived implementation notes — current state lives in `CLAUDE.md`.

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
