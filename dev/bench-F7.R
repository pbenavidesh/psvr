## F7 benchmark: block-k=4 SMO inner loop (Theorem 7 of arXiv:2605.01446 v3).
##
## Part A — R1..R4 regime suite per the plan's acceptance criteria:
##   R1  N=1000  rho_y ~ 2000  RBF sigma = 1     accept >=  5% iter reduction (T3+T7 vs T3)
##   R2  N=1000  rho_y <    5  RBF sigma = 1     accept >= 15% iter reduction (T7 standalone)
##   R3  N=300   rho_y ~ 1000  RBF sigma = 3     accept >= 20% iter reduction (T7 dense)
##   R4  N=1000  rho_y ~ 1000  RBF sigma = 0.3   accept >=  5% iter reduction (T7 sparse)
##
## Part B — T5 (warm-start) x T7 (block-k=4) stacking interaction (paper TODO #10
## scope). Three configurations at N=300, 10-fold CV:
##   B1  F4 cold + F5 warm        (paper baseline)
##   B2  F7 cold + no warm        (T7 alone)
##   B3  F7 cold + F5 warm        (T5+T7 stacked; the current default)
##
## Usage:
##   Rscript dev/bench-F7.R [output.rds]

args     <- commandArgs(trailingOnly = TRUE)
out_path <- if (length(args) >= 1L) args[[1]] else "dev/bench_F7.rds"

suppressPackageStartupMessages({
  devtools::load_all(".", quiet = TRUE)
  library(rsample)
})

REPS <- 5L

## ---- regime fixture builder -------------------------------------------------

make_fixture <- function(N, sdlog, seed = 2026L) {
  set.seed(seed + N + as.integer(round(sdlog * 1000)))
  X <- matrix(rnorm(N * 5), N, 5)
  y <- rlnorm(N, meanlog = 0, sdlog = sdlog)
  list(X = X, y = y, rho_y = max(y) / min(y))
}

## ---- single-fit timer -------------------------------------------------------

time_fit <- function(X, y, kernel, block_k4_enabled, reps = REPS) {
  # Pre-warm to remove first-call dispatch overhead
  invisible(psvr(X[1:20, ], y[1:20], loss = "mape", kernel = kernel,
                 C = 10, eps = 5, block_k4_enabled = block_k4_enabled))

  res <- replicate(reps, {
    gc(reset = TRUE, verbose = FALSE)
    t0  <- Sys.time()
    fit <- psvr(X, y, loss = "mape", kernel = kernel, C = 10, eps = 5,
                block_k4_enabled = block_k4_enabled)
    el  <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
    m   <- fit$solver_meta
    list(elapsed = el, iters = m$iters, converged = m$converged,
         joint   = m$joint_updates,
         k2      = m$k2_fallbacks,
         dr      = m$decoupling_rate,
         dr_e    = m$early_phase_decoupling_rate,
         dr_l    = m$late_phase_decoupling_rate)
  }, simplify = FALSE)

  list(
    wall_med  = median(vapply(res, `[[`, numeric(1), "elapsed")),
    wall_min  = min   (vapply(res, `[[`, numeric(1), "elapsed")),
    iters     = res[[1L]]$iters,
    converged = res[[1L]]$converged,
    joint     = res[[1L]]$joint,
    k2        = res[[1L]]$k2,
    dr        = res[[1L]]$dr,
    dr_e      = res[[1L]]$dr_e,
    dr_l      = res[[1L]]$dr_l
  )
}

## ---- Part A: R1-R4 regimes --------------------------------------------------

REGIMES <- list(
  R1 = list(N = 1000L, sdlog = 1.7,  sigma = 1.0,  target = 0.05,
            label = "heterogeneous + medium-RBF (T3+T7 vs T3)"),
  R2 = list(N = 1000L, sdlog = 0.20, sigma = 1.0,  target = 0.15,
            label = "homogeneous + medium-RBF (T7 standalone)"),
  R3 = list(N = 300L,  sdlog = 1.5,  sigma = 3.0,  target = 0.20,
            label = "heterogeneous + dense-RBF (T7 dense)"),
  R4 = list(N = 1000L, sdlog = 1.4,  sigma = 0.3,  target = 0.05,
            label = "heterogeneous + sparse-RBF (T7 sparse)")
)

verdict <- function(reduction, target) {
  if (reduction >= target)        "PASS"
  else if (reduction >= target/2) "PARTIAL"
  else                             "FAIL"
}

regime_res <- list()

cat(sprintf("=== Part A: R1-R4 regime suite (reps = %d) ===\n", REPS))
for (rn in names(REGIMES)) {
  r <- REGIMES[[rn]]
  fx <- make_fixture(r$N, r$sdlog)
  K  <- make_kernel("rbf", sigma = r$sigma)

  cat(sprintf("\n[%s] %s\n", rn, r$label))
  cat(sprintf("  N=%d  rho_y=%.2f  sigma=%g  target>=%.0f%% iter reduction\n",
              r$N, fx$rho_y, r$sigma, 100 * r$target))

  off <- time_fit(fx$X, fx$y, K, block_k4_enabled = FALSE)
  on  <- time_fit(fx$X, fx$y, K, block_k4_enabled = TRUE)

  iter_red <- 1 - on$iters / off$iters
  wall_red <- 1 - on$wall_med / off$wall_med
  v        <- verdict(iter_red, r$target)

  cat(sprintf("  F4 (off):  iters=%6d  wall=%6.3fs  converged=%s\n",
              off$iters, off$wall_med, off$converged))
  cat(sprintf("  F7 (on):   iters=%6d  wall=%6.3fs  converged=%s  joint=%d  k2=%d  dr=%.3f (early %.3f / late %.3f)\n",
              on$iters, on$wall_med, on$converged, on$joint, on$k2,
              ifelse(is.na(on$dr), NA, on$dr),
              ifelse(is.na(on$dr_e), NA, on$dr_e),
              ifelse(is.na(on$dr_l), NA, on$dr_l)))
  cat(sprintf("  Reduction: iter %+.1f%%   wall %+.1f%%   -> %s\n",
              100 * iter_red, 100 * wall_red, v))

  regime_res[[rn]] <- list(
    rn        = rn,
    N         = r$N,
    rho_y     = fx$rho_y,
    sigma     = r$sigma,
    target    = r$target,
    off       = off,
    on        = on,
    iter_red  = iter_red,
    wall_red  = wall_red,
    verdict   = v
  )
}

## ---- Part B: T5 x T7 stacking interaction at N=300 -------------------------

cat("\n=== Part B: T5 (warm-start) x T7 (block-k=4) stacking, N=300, 10-fold CV ===\n")

set.seed(2026)
N_B  <- 300L
sdlB <- 1.5
d_B  <- data.frame(
  y = rlnorm(N_B, sdlog = sdlB),
  matrix(rnorm(N_B * 5), N_B, 5,
         dimnames = list(NULL, paste0("x", 1:5)))
)
cat(sprintf("  N=%d  rho_y=%.2f  v=10\n", N_B,
            max(d_B$y) / min(d_B$y)))

folds_B <- vfold_cv(d_B, v = 10L)
K_B     <- make_kernel("rbf", sigma = 1)

time_cv <- function(folds, kernel, warm_start, block_k4_enabled, reps = 3L) {
  results <- replicate(reps, {
    gc(reset = TRUE, verbose = FALSE)
    t0  <- Sys.time()
    res <- suppressWarnings(suppressMessages(
      psvr_cv(folds, X_var = paste0("x", 1:5), y_var = "y",
              loss = "mape", kernel = kernel, C = 10, eps = 5,
              warm_start = warm_start,
              block_k4_enabled = block_k4_enabled)
    ))
    el  <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
    list(elapsed = el, iter_count = res$iter_count)
  }, simplify = FALSE)
  list(
    wall_med   = median(vapply(results, `[[`, numeric(1), "elapsed")),
    wall_min   = min   (vapply(results, `[[`, numeric(1), "elapsed")),
    iter_count = results[[1L]]$iter_count,
    iter_sum   = sum(results[[1L]]$iter_count)
  )
}

cat("\n  B1: F4 cold + F5 warm  (block_k4 = FALSE, warm_start = TRUE)\n")
B1 <- time_cv(folds_B, K_B, warm_start = TRUE,  block_k4_enabled = FALSE)
cat(sprintf("    wall %.3f s  iter_sum=%d  per-fold=%s\n",
            B1$wall_med, B1$iter_sum, paste(B1$iter_count, collapse = " ")))

cat("\n  B2: F7 cold + no warm  (block_k4 = TRUE, warm_start = FALSE)\n")
B2 <- time_cv(folds_B, K_B, warm_start = FALSE, block_k4_enabled = TRUE)
cat(sprintf("    wall %.3f s  iter_sum=%d  per-fold=%s\n",
            B2$wall_med, B2$iter_sum, paste(B2$iter_count, collapse = " ")))

cat("\n  B3: F7 cold + F5 warm  (block_k4 = TRUE, warm_start = TRUE; default)\n")
B3 <- time_cv(folds_B, K_B, warm_start = TRUE,  block_k4_enabled = TRUE)
cat(sprintf("    wall %.3f s  iter_sum=%d  per-fold=%s\n",
            B3$wall_med, B3$iter_sum, paste(B3$iter_count, collapse = " ")))

cat("\n  Stacking analysis (relative to B1 = F4+F5):\n")
cat(sprintf("    B2 (T7 alone)        vs B1: wall %+.1f%%   iter_sum %+.1f%%\n",
            100 * (1 - B2$wall_med / B1$wall_med),
            100 * (1 - B2$iter_sum / B1$iter_sum)))
cat(sprintf("    B3 (T5+T7 stacked)   vs B1: wall %+.1f%%   iter_sum %+.1f%%\n",
            100 * (1 - B3$wall_med / B1$wall_med),
            100 * (1 - B3$iter_sum / B1$iter_sum)))
cat(sprintf("    B3 vs B2 (warm gain over T7-alone): wall %+.1f%%   iter_sum %+.1f%%\n",
            100 * (1 - B3$wall_med / B2$wall_med),
            100 * (1 - B3$iter_sum / B2$iter_sum)))

## ---- Final summary ---------------------------------------------------------

cat("\n=== Summary table ===\n")
cat(sprintf("%-4s | %5s | %8s | %5s | %7s | %7s | %7s | %7s | %5s | %5s | %s\n",
            "rgme", "N", "rho_y", "sigma", "F4iter", "F7iter", "iter%", "wall%", "drO", "drL", "verdict"))
cat(strrep("-", 100), "\n", sep = "")
for (rn in names(regime_res)) {
  r <- regime_res[[rn]]
  cat(sprintf("%-4s | %5d | %8.1f | %5.2f | %7d | %7d | %+6.1f%% | %+6.1f%% | %5.3f | %5.3f | %s\n",
              r$rn, r$N, r$rho_y, r$sigma, r$off$iters, r$on$iters,
              100 * r$iter_red, 100 * r$wall_red,
              ifelse(is.na(r$on$dr),   NA, r$on$dr),
              ifelse(is.na(r$on$dr_l), NA, r$on$dr_l),
              r$verdict))
}

results <- list(
  regimes = regime_res,
  stacking = list(B1 = B1, B2 = B2, B3 = B3,
                  N = N_B, sdlog = sdlB, rho_y = max(d_B$y) / min(d_B$y)),
  meta = list(
    date     = Sys.time(),
    reps     = REPS,
    git_head = tryCatch(system("git rev-parse HEAD", intern = TRUE),
                        error = function(e) NA_character_)
  )
)
saveRDS(results, out_path)
cat(sprintf("\nSaved to: %s\n", out_path))
