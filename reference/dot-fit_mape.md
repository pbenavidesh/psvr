# Fit epsilon-SVR with MAPE loss (Model 1) — internal

Internal fitter for the MAPE epsilon-SVR family. Use
[`psvr_mape()`](https://pbenavidesh.github.io/psvr/reference/psvr_mape.md)
instead. Returns the `psvr_mape` shape, which is what
[`psvr_mape()`](https://pbenavidesh.github.io/psvr/reference/psvr_mape.md)
and the parsnip engine fit wrappers both return.

## Usage

``` r
.fit_mape(
  X,
  y,
  kernel,
  C,
  eps,
  solver = c("smo", "osqp"),
  tol = 0.001,
  max_iter = 100000L,
  alpha_init = NULL,
  alpha_star_init = NULL,
  warm_start_check = TRUE,
  precomputed_Omega = NULL,
  block_k4_enabled = TRUE,
  alpha_couple = 0.5,
  engine = c("rcpp", "r")
)
```

## Arguments

- X, y, kernel, C, eps, solver, tol:

  See
  [`psvr_mape()`](https://pbenavidesh.github.io/psvr/reference/psvr_mape.md).

- alpha_init, alpha_star_init:

  Optional length-N numeric warm-start vectors (Theorem 5); `NULL`
  cold-starts.

- warm_start_check:

  Logical; if `TRUE`, validate the post-projection feasibility of the
  warm-start vectors. Default `TRUE`.

## Value

A list of class `"psvr_mape"` (legacy shape).
