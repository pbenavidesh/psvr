# Fit symmetric epsilon-SVR with MAPE loss (Model 2) — internal

Internal fitter for the symmetric MAPE epsilon-SVR family. Use
[`psvr_mape()`](https://pbenavidesh.github.io/psvr/reference/psvr_mape.md)
with `sym_type = "even"` / `"odd"` instead. Returns the `psvr_mape_sym`
shape, which is what
[`psvr_mape()`](https://pbenavidesh.github.io/psvr/reference/psvr_mape.md)
and the parsnip engine fit wrappers both return with `sym_type = "even"`
/ `"odd"`. The kernel must satisfy Assumption 3 of the paper; see
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
  precomputed_Omega_s = NULL,
  block_k4_enabled = TRUE,
  alpha_couple = 0.5,
  engine = c("rcpp", "r")
)
```

## Arguments

- X, y, kernel, C, eps, solver, tol:

  See
  [`psvr_mape()`](https://pbenavidesh.github.io/psvr/reference/psvr_mape.md).

- a:

  Symmetry type: `1` (even) or `-1` (odd). This is the internal integer;
  the public argument is `sym_type`, with `"even"` mapping to `a = 1`
  and `"odd"` to `a = -1`. Neither
  [`psvr_mape()`](https://pbenavidesh.github.io/psvr/reference/psvr_mape.md)
  nor the parsnip specifications expose `a` directly.

- alpha_init, alpha_star_init:

  Optional length-N numeric warm-start vectors (Theorem 5); `NULL`
  cold-starts.

- warm_start_check:

  Logical; if `TRUE`, validate the post-projection feasibility of the
  warm-start vectors. Default `TRUE`.

## Value

A list of class `"psvr_mape_sym"` (legacy shape).
