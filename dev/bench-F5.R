## F5 benchmark: T5 (warm-start) effect on 10-fold cross-validation.
##
## T4 was reverted to Fan-Chen-Lin WSS3 after empirical evaluation showed
## that the Glasmachers-Igel max-gain alternative produced identical iter
## counts on our fixtures (the analytic step typically fits the per-sample
## box without clipping).  T4 single-fit bench is therefore moot.
##
## This script reports the T5 cumulative speedup at two scales:
##   * N = 300  (the canonical F5 fixture)
##   * N = 1000 (calibration check whether the speedup grows with N)
##
## Usage:
##   Rscript dev/bench-F5.R <output.rds>

args <- commandArgs(trailingOnly = TRUE)
out_path <- if (length(args) >= 1L) args[[1]] else "dev/bench_F5.rds"

suppressPackageStartupMessages({
  devtools::load_all(".")
  library(rsample)
})

run_cv_bench <- function(N, label) {
  set.seed(2026)
  d <- data.frame(
    y = stats::rlnorm(N, sdlog = 1.5),
    matrix(stats::rnorm(N * 5), N, 5,
           dimnames = list(NULL, paste0("x", 1:5)))
  )
  folds <- vfold_cv(d, v = 10L)
  X_var <- paste0("x", 1:5); y_var <- "y"
  K     <- make_kernel("rbf", sigma = 1)

  cat(sprintf("\n[T5 bench %s] N = %d  folds = 10  rho_y = %.0f\n",
              label, N, max(d$y) / min(d$y)))

  t_cold <- system.time(
    res_cold <- suppressWarnings(
      psvr_cv(folds, X_var = X_var, y_var = y_var,
              loss = "mape", kernel = K, C = 10, eps = 5,
              warm_start = FALSE)
    )
  )["elapsed"]
  t_warm <- system.time(
    res_warm <- suppressWarnings(
      psvr_cv(folds, X_var = X_var, y_var = y_var,
              loss = "mape", kernel = K, C = 10, eps = 5,
              warm_start = TRUE)
    )
  )["elapsed"]

  cat(sprintf("  Cold-start: elapsed %.3f s   total iters = %d\n",
              as.numeric(t_cold), sum(res_cold$iter_count)))
  cat(sprintf("  Warm-start: elapsed %.3f s   total iters = %d\n",
              as.numeric(t_warm), sum(res_warm$iter_count)))
  cat(sprintf("  Iter speedup (1 - warm/cold): %.1f%%\n",
              100 * (1 - sum(res_warm$iter_count) / sum(res_cold$iter_count))))
  cat(sprintf("  Wall speedup (cold/warm):     %.2fx\n",
              as.numeric(t_cold) / as.numeric(t_warm)))

  list(N = N,
       t_cold = as.numeric(t_cold), t_warm = as.numeric(t_warm),
       iters_cold = res_cold$iter_count,
       iters_warm = res_warm$iter_count)
}

results <- list()
results[["N=300"]]  <- run_cv_bench(300L,  "canonical")
results[["N=1000"]] <- run_cv_bench(1000L, "calibration")

saveRDS(results, out_path)
cat(sprintf("\nSaved to: %s\n", out_path))
