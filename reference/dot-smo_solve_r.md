# SMO solver — R reference implementation (engine = "r").

Canonical R-level algorithm. Bit-identical reference for the Rcpp core
in src/core_smo_solve.cpp. Will be deprecated in v0.0.4.0 and removed in
v0.1.0. Do NOT call directly; go through
[`.smo_solve()`](https://pbenavidesh.github.io/psvr/reference/dot-smo_solve.md).

## Usage

``` r
.smo_solve_r(
  K_acc,
  y,
  C,
  eps,
  tol = 0.001,
  max_iter = 100000L,
  n_check = NULL,
  n_freeze = 5L,
  alpha_init = NULL,
  alpha_star_init = NULL,
  warm_start_check = TRUE,
  new_mask = NULL,
  block_k4_enabled = TRUE,
  alpha_couple = 0.5,
  trace = FALSE
)
```
