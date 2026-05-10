## F3 performance sanity check.
## Measures the spectral-shift overhead as a fraction of total Model 2 fit
## time on RBF + a=-1 + N=200, the worst case (a=-1 forces both
## passes to run; RBF is fast so shift overhead is most visible).
## Acceptance: spectral overhead < 15% of total fit time.

suppressPackageStartupMessages(devtools::load_all(
  "C:/Users/behep/OneDrive - ITESO/PhD/00-Tesis/psvr",
  quiet = TRUE
))

set.seed(42)
N <- 200L; p <- 10L
X <- matrix(stats::rnorm(N * p), N, p)
y <- stats::rlnorm(N)
K <- make_kernel("rbf", sigma = 1)

# Warmup (BLAS, JIT, etc.)
invisible(replicate(3, psvr(X, y, loss = "mape", sym = -1L,
                            kernel = K, C = 10, eps = 5)))

n_reps <- 30L

# (1) Total fit time (F3 active).
t_total <- replicate(n_reps, system.time(
  psvr(X, y, loss = "mape", sym = -1L, kernel = K, C = 10, eps = 5)
)["elapsed"])

# (2) Isolated spectral-shift time.
Omega_s <- sym_kernel_matrix(K, X, a = -1L)
diag(Omega_s) <- diag(Omega_s) + 0.5e-6
t_shift <- replicate(n_reps, system.time(
  psvr:::.adaptive_spectral_shift(Omega_s)
)["elapsed"])

# (3) Time of the kernel-matrix construction alone, for context.
t_kmat <- replicate(n_reps, system.time({
  Os <- sym_kernel_matrix(K, X, a = -1L)
  diag(Os) <- diag(Os) + 0.5e-6
})["elapsed"])

med_total <- median(t_total)
med_shift <- median(t_shift)
med_kmat  <- median(t_kmat)
overhead  <- 100 * med_shift / med_total

cat("=== F3 performance benchmark (Model 2, RBF, a=-1, N=200) ===\n")
cat(sprintf("Total fit time   median: %7.4f s   (n=%d)\n", med_total, n_reps))
cat(sprintf("Kernel matrix    median: %7.4f s   (%.2f%% of fit)\n",
            med_kmat, 100 * med_kmat / med_total))
cat(sprintf("Spectral shift   median: %7.4f s   (%.2f%% of fit)\n",
            med_shift, overhead))
cat(sprintf("Acceptance:      < 15%% of fit time?  ", overhead),
    if (overhead < 15) "PASS" else "FAIL", "\n")

cat(sprintf("\nFit time min/max: %.4f / %.4f s\n",
            min(t_total), max(t_total)))
cat(sprintf("Shift time min/max: %.4f / %.4f s\n",
            min(t_shift), max(t_shift)))
