# Fit symmetric epsilon-SVR with MAPE loss (Model 2) — internal

Internal fitter for the symmetric MAPE epsilon-SVR family. Use
[`psvr()`](https://pbenavidesh.github.io/psvr/reference/psvr.md) with
`loss = "mape"` and `sym = +1L` / `-1L` instead. Returns the legacy
`psvr_mape_sym` shape; the deprecation wrapper
[`mape_sym_svr()`](https://pbenavidesh.github.io/psvr/reference/mape_sym_svr.md)
forwards directly to this function. The kernel must satisfy Assumption 3
of the paper; see
[`make_kernel()`](https://pbenavidesh.github.io/psvr/reference/make_kernel.md).

## Usage

``` r
.fit_mape_sym(
  X,
  y,
  kernel,
  C,
  eps,
  a = 1,
  solver = c("smo", "osqp"),
  tol = 0.001,
  max_iter = 100000L,
  alpha_init = NULL,
  alpha_star_init = NULL,
  warm_start_check = TRUE,
  new_mask = NULL,
  precomputed_Omega_s = NULL,
  block_k4_enabled = TRUE,
  alpha_couple = 0.5,
  engine = c("rcpp", "r")
)
```

## Arguments

- X, y, kernel, C, eps, a, solver, tol:

  See
  [`mape_sym_svr()`](https://pbenavidesh.github.io/psvr/reference/mape_sym_svr.md).

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

A list of class `"psvr_mape_sym"` (legacy shape).
