# Dispatcher: SMO solver with engine choice (R reference vs Rcpp core).

Forwards to the F7-C-full Rcpp core (`engine = "rcpp"`, default) or the
R reference implementation (`engine = "r"`). The R path is the canonical
algorithm and remains the bit-identical reference for the Rcpp port; it
will be deprecated in v0.0.4.0 and removed in v0.1.0 once the Rcpp path
has passed the snapshot and engine-equivalence tests for at least two
release cycles.

## Usage

``` r
.smo_solve(
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
  block_k4_enabled = TRUE,
  alpha_couple = 0.5,
  trace = FALSE,
  engine = c("rcpp", "r")
)
```

## Details

Warm-start projection (Algorithm 1) runs in R via `.warm_start_init()`
BEFORE the core call, regardless of engine, so both paths see
already-feasible alpha/alpha\* on entry.
