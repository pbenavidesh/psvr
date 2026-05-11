## F6 benchmark: Rcpp-accelerated kernel_matrix + cross-fold reuse in
## psvr_cv. Targets vs F5 baseline (from F5 profile at N=1000):
##   - kernel_matrix(N=1000, RBF):   730 ms / 222 MB  -> target  <= 100 ms
##   - psvr() end-to-end (N=1000):   820 ms           -> target  <= 250 ms
##   - psvr_cv() 10-fold (N=1000):  ~8 s              -> target  <= 1.5 s
##
## Usage:
##   Rscript dev/bench-F6.R <output.rds>

args <- commandArgs(trailingOnly = TRUE)
out_path <- if (length(args) >= 1L) args[[1]] else "dev/bench_F6.rds"

suppressPackageStartupMessages({
  devtools::load_all(".", quiet = TRUE)
  library(rsample)
})

set.seed(2026)

bench_kernel <- function(N) {
  set.seed(2026 + N)
  X <- matrix(rnorm(N * 5), N, 5)
  K <- make_kernel("rbf", sigma = 1)

  # Pre-warm (load DLL + R-side dispatch overhead)
  invisible(psvr:::kernel_matrix(K, X[1:10, , drop = FALSE]))

  # Three reps, take median wall time
  reps <- replicate(3L, {
    gc(reset = TRUE, verbose = FALSE)
    t0 <- Sys.time()
    M <- psvr:::kernel_matrix(K, X)
    as.numeric(difftime(Sys.time(), t0, units = "secs"))
  })
  list(N = N, wall_s = median(reps), wall_min_s = min(reps), wall_max_s = max(reps),
       size_MB = as.numeric(object.size(psvr:::kernel_matrix(K, X))) / 1024^2)
}

bench_psvr_e2e <- function(N) {
  set.seed(2026 + N)
  X <- matrix(rnorm(N * 5), N, 5)
  y <- rlnorm(N, sdlog = 1.5)
  K <- make_kernel("rbf", sigma = 1)

  invisible(psvr(X[1:20, ], y[1:20], loss = "mape", kernel = K, C = 10, eps = 5))

  reps <- replicate(3L, {
    gc(reset = TRUE, verbose = FALSE)
    t0  <- Sys.time()
    fit <- psvr(X, y, loss = "mape", kernel = K, C = 10, eps = 5)
    list(elapsed = as.numeric(difftime(Sys.time(), t0, units = "secs")),
         iters   = fit$solver_meta$iters)
  }, simplify = FALSE)
  walls <- vapply(reps, `[[`, numeric(1), "elapsed")
  list(N = N, wall_s = median(walls), wall_min_s = min(walls), wall_max_s = max(walls),
       iters = reps[[1L]]$iters)
}

bench_psvr_cv <- function(N, v = 10L) {
  set.seed(2026)
  d <- data.frame(
    y  = rlnorm(N, sdlog = 1.5),
    matrix(rnorm(N * 5), N, 5,
           dimnames = list(NULL, paste0("x", 1:5)))
  )
  folds <- vfold_cv(d, v = v)
  K     <- make_kernel("rbf", sigma = 1)

  # With cross-fold reuse (F6 default for rset inputs)
  t_reuse <- system.time({
    res_reuse <- suppressWarnings(psvr_cv(folds, X_var = paste0("x", 1:5), y_var = "y",
                                          loss = "mape", kernel = K, C = 10, eps = 5,
                                          warm_start = TRUE, verbose = FALSE))
  })["elapsed"]

  # Without cross-fold reuse: force the list-of-tuples path (no precompute).
  folds_list <- lapply(folds$splits, function(s) {
    list(analysis   = rsample::analysis(s),
         assessment = rsample::assessment(s),
         row_ids    = s$in_id)
  })
  t_no_reuse <- system.time({
    res_no_reuse <- suppressWarnings(psvr_cv(folds_list,
                                             X_var = paste0("x", 1:5), y_var = "y",
                                             loss = "mape", kernel = K, C = 10, eps = 5,
                                             warm_start = TRUE, verbose = FALSE))
  })["elapsed"]

  list(N = N, v = v,
       wall_reuse_s    = as.numeric(t_reuse),
       wall_no_reuse_s = as.numeric(t_no_reuse),
       iters_reuse     = sum(res_reuse$iter_count),
       iters_no_reuse  = sum(res_no_reuse$iter_count))
}

cat("\n=== 6.1 kernel_matrix alone (RBF) ===\n")
kernel_res <- list()
for (N in c(300L, 1000L, 3000L, 10000L)) {
  kernel_res[[paste0("N=", N)]] <- r <- bench_kernel(N)
  cat(sprintf("  N=%5d  wall median = %.3f s  (min %.3f / max %.3f)   size = %.1f MB\n",
              r$N, r$wall_s, r$wall_min_s, r$wall_max_s, r$size_MB))
}

cat("\n=== 6.2 psvr() end-to-end (RBF, MAPE) ===\n")
e2e_res <- list()
for (N in c(300L, 1000L)) {
  e2e_res[[paste0("N=", N)]] <- r <- bench_psvr_e2e(N)
  cat(sprintf("  N=%5d  wall median = %.3f s  (min %.3f / max %.3f)   iters = %d\n",
              r$N, r$wall_s, r$wall_min_s, r$wall_max_s, r$iters))
}

cat("\n=== 6.3 psvr_cv() 10-fold (RBF, MAPE, warm-start) ===\n")
cv_res <- list()
for (N in c(300L, 1000L)) {
  cv_res[[paste0("N=", N)]] <- r <- bench_psvr_cv(N, v = 10L)
  speedup <- r$wall_no_reuse_s / r$wall_reuse_s
  cat(sprintf("  N=%5d  reuse %.3f s   no-reuse %.3f s   reuse speedup %.2fx   iters %d == %d\n",
              r$N, r$wall_reuse_s, r$wall_no_reuse_s, speedup,
              r$iters_reuse, r$iters_no_reuse))
}

results <- list(
  kernel = kernel_res,
  e2e    = e2e_res,
  cv     = cv_res,
  meta   = list(date = Sys.time(), git_head = system("git rev-parse HEAD", intern = TRUE))
)
saveRDS(results, out_path)
cat(sprintf("\nSaved to: %s\n", out_path))
