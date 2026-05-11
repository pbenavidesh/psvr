## F7 benchmark: block-k=4 SMO inner loop + C-full portable architecture.
##
## Part A — R1..R4 regime suite × {engine = "r", "rcpp"} × {block_k4 OFF, ON}:
##   R1  N=1000  rho_y ~ 2000  RBF sigma = 1     accept >=  5% iter reduction
##   R2  N=1000  rho_y <    5  RBF sigma = 1     accept >= 15%
##   R3  N=300   rho_y ~ 1000  RBF sigma = 3     accept >= 20%
##   R4  N=1000  rho_y ~ 1000  RBF sigma = 0.3   accept >=  5%
##
## Part B — T5 (warm-start) x T7 (block-k=4) stacking × 2 engines at N=300,
## 10-fold CV (paper TODO #10 scope):
##   B1  cold + F5 warm        (block_k4 = FALSE, warm_start = TRUE)
##   B2  F7  + no warm         (block_k4 = TRUE,  warm_start = FALSE)
##   B3  F7  + F5 warm         (block_k4 = TRUE,  warm_start = TRUE; default)
## each under engine = "r" and engine = "rcpp".
##
## C-full claim being tested:
##   F4-R    -> F4-Rcpp:    3-5x wall reduction (per-iter scaffolding eliminated)
##   F7-R    -> F7-Rcpp:    4-7x wall reduction
##   F7-Rcpp vs F4-Rcpp:    10-30% wall positive (T7 paper TODO #9 resolved)
##   CV B3-rcpp vs B1-r:    5-9x combined wall reduction
##
## Usage:
##   Rscript dev/bench-F7.R [output.rds]
##
## Note: pause OneDrive sync during this run (per F5/F6 lesson — avoids
## file-lock interruptions on long runs).

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

time_fit <- function(X, y, kernel, engine, block_k4_enabled, reps = REPS) {
  # Pre-warm (avoid first-call dispatch overhead in the timing).
  invisible(suppressWarnings(
    psvr(X[1:20, ], y[1:20], loss = "mape", kernel = kernel,
         C = 10, eps = 5, engine = engine,
         block_k4_enabled = block_k4_enabled)
  ))

  res <- replicate(reps, {
    gc(reset = TRUE, verbose = FALSE)
    t0  <- Sys.time()
    fit <- suppressWarnings(
      psvr(X, y, loss = "mape", kernel = kernel, C = 10, eps = 5,
           engine = engine, block_k4_enabled = block_k4_enabled)
    )
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
    engine    = engine,
    block_k4  = block_k4_enabled,
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

## ---- Part A: R1-R4 × {r, rcpp} × {F4, F7} ---------------------------------

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

cat(sprintf("=== Part A: R1-R4 regime suite (reps = %d, 4 configs per regime) ===\n", REPS))
for (rn in names(REGIMES)) {
  r <- REGIMES[[rn]]
  fx <- make_fixture(r$N, r$sdlog)
  K  <- make_kernel("rbf", sigma = r$sigma)

  cat(sprintf("\n[%s] %s\n", rn, r$label))
  cat(sprintf("  N=%d  rho_y=%.2f  sigma=%g  target>=%.0f%% iter reduction\n",
              r$N, fx$rho_y, r$sigma, 100 * r$target))

  # Run all 4 (engine, mode) combinations
  r_F4    <- time_fit(fx$X, fx$y, K, engine = "r",    block_k4_enabled = FALSE)
  rcpp_F4 <- time_fit(fx$X, fx$y, K, engine = "rcpp", block_k4_enabled = FALSE)
  r_F7    <- time_fit(fx$X, fx$y, K, engine = "r",    block_k4_enabled = TRUE)
  rcpp_F7 <- time_fit(fx$X, fx$y, K, engine = "rcpp", block_k4_enabled = TRUE)

  iter_red_r    <- 1 - r_F7$iters    / r_F4$iters
  iter_red_rcpp <- 1 - rcpp_F7$iters / rcpp_F4$iters
  wall_red_r    <- 1 - r_F7$wall_med    / r_F4$wall_med
  wall_red_rcpp <- 1 - rcpp_F7$wall_med / rcpp_F4$wall_med
  speedup_F4    <- r_F4$wall_med / rcpp_F4$wall_med  # F4 R → F4 Rcpp
  speedup_F7    <- r_F7$wall_med / rcpp_F7$wall_med  # F7 R → F7 Rcpp
  rcpp_F7_vs_F4 <- 1 - rcpp_F7$wall_med / rcpp_F4$wall_med
  v_iter <- verdict(iter_red_rcpp, r$target)

  cat(sprintf("  F4 R   :  iters=%6d  wall=%7.3fs   converged=%s\n",
              r_F4$iters,    r_F4$wall_med, r_F4$converged))
  cat(sprintf("  F4 Rcpp:  iters=%6d  wall=%7.3fs   converged=%s   F4 speedup R->Rcpp: %.2fx\n",
              rcpp_F4$iters, rcpp_F4$wall_med, rcpp_F4$converged, speedup_F4))
  cat(sprintf("  F7 R   :  iters=%6d  wall=%7.3fs   converged=%s   joint=%d  dr=%.3f (e %.3f / l %.3f)\n",
              r_F7$iters, r_F7$wall_med, r_F7$converged,
              r_F7$joint, ifelse(is.na(r_F7$dr), NA, r_F7$dr),
              ifelse(is.na(r_F7$dr_e), NA, r_F7$dr_e),
              ifelse(is.na(r_F7$dr_l), NA, r_F7$dr_l)))
  cat(sprintf("  F7 Rcpp:  iters=%6d  wall=%7.3fs   converged=%s   joint=%d   F7 speedup R->Rcpp: %.2fx\n",
              rcpp_F7$iters, rcpp_F7$wall_med, rcpp_F7$converged,
              rcpp_F7$joint, speedup_F7))
  cat(sprintf("  F7-Rcpp vs F4-Rcpp:  iter %+.1f%%   wall %+.1f%%   ->  %s\n",
              100 * iter_red_rcpp, 100 * rcpp_F7_vs_F4, v_iter))

  regime_res[[rn]] <- list(
    rn        = rn,
    N         = r$N,
    rho_y     = fx$rho_y,
    sigma     = r$sigma,
    target    = r$target,
    r_F4      = r_F4,
    rcpp_F4   = rcpp_F4,
    r_F7      = r_F7,
    rcpp_F7   = rcpp_F7,
    iter_red_r    = iter_red_r,
    iter_red_rcpp = iter_red_rcpp,
    wall_red_r    = wall_red_r,
    wall_red_rcpp = wall_red_rcpp,
    speedup_F4    = speedup_F4,
    speedup_F7    = speedup_F7,
    rcpp_F7_vs_F4 = rcpp_F7_vs_F4,
    verdict       = v_iter
  )
}

## ---- Part B: T5 x T7 stacking × 2 engines, N=300, 10-fold CV ---------------

cat("\n=== Part B: T5 x T7 stacking × 2 engines, N=300, 10-fold CV ===\n")

set.seed(2026)
N_B  <- 300L
sdlB <- 1.5
d_B  <- data.frame(
  y = rlnorm(N_B, sdlog = sdlB),
  matrix(rnorm(N_B * 5), N_B, 5,
         dimnames = list(NULL, paste0("x", 1:5)))
)
cat(sprintf("  N=%d  rho_y=%.2f  v=10\n", N_B, max(d_B$y) / min(d_B$y)))

folds_B <- vfold_cv(d_B, v = 10L)
K_B     <- make_kernel("rbf", sigma = 1)

time_cv <- function(folds, kernel, warm_start, block_k4_enabled, engine,
                    reps = 3L) {
  results <- replicate(reps, {
    gc(reset = TRUE, verbose = FALSE)
    t0  <- Sys.time()
    res <- suppressWarnings(suppressMessages(
      psvr_cv(folds, X_var = paste0("x", 1:5), y_var = "y",
              loss = "mape", kernel = kernel, C = 10, eps = 5,
              warm_start = warm_start,
              block_k4_enabled = block_k4_enabled,
              engine = engine)
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

# 6 configs
B_specs <- list(
  list(id = "B1-r",    eng = "r",    warm = TRUE,  bk4 = FALSE,
       label = "F4 cold + F5 warm  (paper baseline)"),
  list(id = "B1-rcpp", eng = "rcpp", warm = TRUE,  bk4 = FALSE,
       label = "F4 cold + F5 warm  (Rcpp)"),
  list(id = "B2-r",    eng = "r",    warm = FALSE, bk4 = TRUE,
       label = "F7 + no warm       (T7 alone, R)"),
  list(id = "B2-rcpp", eng = "rcpp", warm = FALSE, bk4 = TRUE,
       label = "F7 + no warm       (T7 alone, Rcpp)"),
  list(id = "B3-r",    eng = "r",    warm = TRUE,  bk4 = TRUE,
       label = "F7 + F5 warm       (stacked, R)"),
  list(id = "B3-rcpp", eng = "rcpp", warm = TRUE,  bk4 = TRUE,
       label = "F7 + F5 warm       (stacked, Rcpp; default)")
)

B_res <- list()
for (s in B_specs) {
  cat(sprintf("\n  %-8s %-50s\n", s$id, s$label))
  r <- time_cv(folds_B, K_B, warm_start = s$warm,
               block_k4_enabled = s$bk4, engine = s$eng)
  cat(sprintf("    wall %6.3f s   iter_sum=%6d   per-fold=%s\n",
              r$wall_med, r$iter_sum, paste(r$iter_count, collapse = " ")))
  B_res[[s$id]] <- c(s, r)
}

baseline <- B_res[["B1-r"]]
cat(sprintf("\n  Speedup vs B1-r (F4+F5 warm, R baseline = paper claim) [wall_med = %.3f s]:\n",
            baseline$wall_med))
for (s in B_specs) {
  br <- B_res[[s$id]]
  sp <- baseline$wall_med / br$wall_med
  itr <- 1 - br$iter_sum / baseline$iter_sum
  cat(sprintf("    %-8s wall=%6.3fs   speedup=%5.2fx   iter_sum %+6.1f%%\n",
              s$id, br$wall_med, sp, 100 * itr))
}

## ---- Final summary tables -------------------------------------------------

cat("\n=== Summary A: R1-R4 (engine × mode) ===\n")
cat(sprintf("%-4s | %5s | %8s | %5s | %-7s | %-5s | %7s | %8s | %s\n",
            "rgme", "N", "rho_y", "sig", "engine", "mode", "iters", "wall_s", "dr"))
cat(strrep("-", 78), "\n", sep = "")
for (rn in names(regime_res)) {
  rr <- regime_res[[rn]]
  for (which_eng in c("r", "rcpp")) {
    for (mode in c("F4", "F7")) {
      key <- sprintf("%s_%s", which_eng, mode)
      cfg <- rr[[key]]
      cat(sprintf("%-4s | %5d | %8.1f | %5.2f | %-7s | %-5s | %7d | %8.3f | %s\n",
                  rn, rr$N, rr$rho_y, rr$sigma, which_eng, mode,
                  cfg$iters, cfg$wall_med,
                  if (mode == "F4") "-" else
                    sprintf("%.3f", ifelse(is.na(cfg$dr), 0, cfg$dr))))
    }
  }
}

cat("\n=== Summary A reduction table (Rcpp speedups + F7 vs F4 wall) ===\n")
cat(sprintf("%-4s | %7s | %7s | %12s | %12s | %s\n",
            "rgme", "F4 R->C", "F7 R->C", "F7 iter Δ", "F7 wall Δ", "verdict"))
cat(strrep("-", 70), "\n", sep = "")
for (rn in names(regime_res)) {
  rr <- regime_res[[rn]]
  cat(sprintf("%-4s | %5.2fx  | %5.2fx  |   %+6.1f%%   |   %+6.1f%%   | %s\n",
              rn, rr$speedup_F4, rr$speedup_F7,
              100 * rr$iter_red_rcpp,
              100 * rr$rcpp_F7_vs_F4,
              rr$verdict))
}

cat("\n=== Summary B: T5 x T7 stacking (CV at N=300) ===\n")
cat(sprintf("%-8s | %-6s | %5s | %5s | %8s | %8s | %s\n",
            "id", "engine", "warm", "bk4", "wall_s", "iter_sum", "speedup vs B1-r"))
cat(strrep("-", 78), "\n", sep = "")
for (s in B_specs) {
  br <- B_res[[s$id]]
  cat(sprintf("%-8s | %-6s | %5s | %5s | %8.3f | %8d | %5.2fx\n",
              s$id, s$eng, s$warm, s$bk4,
              br$wall_med, br$iter_sum,
              baseline$wall_med / br$wall_med))
}

results <- list(
  regimes = regime_res,
  stacking = B_res,
  meta = list(
    date     = Sys.time(),
    reps     = REPS,
    N_B      = N_B,
    rho_y_B  = max(d_B$y) / min(d_B$y),
    git_head = tryCatch(system("git rev-parse HEAD", intern = TRUE),
                        error = function(e) NA_character_)
  )
)
saveRDS(results, out_path)
cat(sprintf("\nSaved to: %s\n", out_path))
