## F4 benchmark: heterogeneous vs homogeneous targets.
##
## Replicates psvr() fits 20 times each on two y vectors with very
## different rho_y, then writes wall-clock medians + iter counts to
## an RDS file for F3-vs-F4 comparison.
##
## Usage:
##   Rscript dev/bench-F4.R <output.rds>

args <- commandArgs(trailingOnly = TRUE)
out_path <- if (length(args) >= 1L) args[[1]] else "dev/bench_capture.rds"

suppressPackageStartupMessages({
  devtools::load_all(".")
})

set.seed(2026)
N <- 200L
X <- matrix(stats::rnorm(N * 10), N, 10)
y_het <- stats::rlnorm(N, sdlog = 1.5)         # heterogeneous: rho_y ~ 2388
y_hom <- rep(2.0, N) + stats::rnorm(N) * 0.05  # homogeneous: rho_y ~ 1.2

K <- make_kernel("rbf", sigma = 1)

run_once <- function(y) {
  t <- system.time(
    fit <- suppressWarnings(
      psvr(X, y, loss = "mape", kernel = K, C = 10, eps = 5)
    )
  )["elapsed"]
  as.numeric(t)
}

# Warm-up (R + JIT settling), discarded.
invisible(run_once(y_het))
invisible(run_once(y_hom))

reps <- 20L
times_het <- replicate(reps, run_once(y_het))
times_hom <- replicate(reps, run_once(y_hom))

# Iter-count for the median run (single-shot via direct .smo_solve call).
ns <- asNamespace("psvr")
get_iters <- function(y) {
  Omega <- ns$kernel_matrix(K, X)
  diag(Omega) <- diag(Omega) + 1e-6
  K_acc <- ns$.make_kernel_accessor(Omega)
  out <- suppressWarnings(ns$.smo_solve(K_acc, y, C = 10, eps = 5))
  list(iters = out$iterations, converged = out$converged)
}
it_het <- get_iters(y_het)
it_hom <- get_iters(y_hom)

cat(sprintf("rho_y(het) = %.1f, rho_y(hom) = %.2f\n",
            max(y_het) / min(y_het), max(y_hom) / min(y_hom)))
cat(sprintf("Heterogeneous: median %.3f s   iters=%d converged=%s\n",
            median(times_het), it_het$iters, it_het$converged))
cat(sprintf("Homogeneous:   median %.3f s   iters=%d converged=%s\n",
            median(times_hom), it_hom$iters, it_hom$converged))

saveRDS(
  list(
    times_het = times_het, times_hom = times_hom,
    it_het = it_het, it_hom = it_hom,
    rho_het = max(y_het) / min(y_het),
    rho_hom = max(y_hom) / min(y_hom)
  ),
  out_path
)
cat(sprintf("Saved to: %s\n", out_path))
