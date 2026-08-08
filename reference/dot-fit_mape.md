# Fit epsilon-SVR with MAPE loss (Model 1) — internal

Internal fitter for the MAPE epsilon-SVR family. Use
[`psvr()`](https://pbenavidesh.github.io/psvr/reference/psvr.md) with
`loss = "mape"` instead. Returns the legacy `psvr_mape` shape; the
deprecation wrapper
[`mape_svr()`](https://pbenavidesh.github.io/psvr/reference/mape_svr.md)
forwards directly to this function.

## Usage

``` r
.fit_mape(
  X,
  y,
  kernel,
  C,
  eps,
  solver = c("smo", "osqp"),
  tol = 1e-05,
  alpha_init = NULL,
  alpha_star_init = NULL,
  warm_start_check = TRUE,
  new_mask = NULL,
  precomputed_Omega = NULL,
  block_k4_enabled = TRUE,
  alpha_couple = 0.5,
  engine = c("rcpp", "r")
)
```

## Arguments

- X, y, kernel, C, eps, solver, tol:

  See
  [`mape_svr()`](https://pbenavidesh.github.io/psvr/reference/mape_svr.md).

- alpha_init, alpha_star_init:

  Optional length-N numeric warm-start vectors (Theorem 5); `NULL`
  cold-starts.

- warm_start_check:

  Logical; if `TRUE`, validate the post-projection feasibility of the
  warm-start vectors. Default `TRUE`.

- new_mask:

  Optional logical vector (length N) flagging samples that are NEW
  relative to the previous fit (used to distribute the equality-
  constraint projection over new samples only). `NULL` infers "new =
  both alpha and alpha_star are exactly zero".

## Value

A list of class `"psvr_mape"` (legacy shape).
