## Capture SMO iter counts + converged flag for each MAPE test case.
##
## Replays the exact Omega-matrix construction performed by the fitters
## (.fit_mape / .fit_mape_sym) and calls .smo_solve() directly so we can
## inspect `iterations` and `converged` from the solver return list.
##
## Usage:
##   Rscript dev/iter_capture.R <output.rds>

args <- commandArgs(trailingOnly = TRUE)
out_path <- if (length(args) >= 1L) args[[1]] else "dev/iters_capture.rds"

suppressPackageStartupMessages({
  devtools::load_all(".")
})

# Snapshot fixture (same as test-bit-identical.R / test-psvr-direct.R)
set.seed(2026)
X <- matrix(stats::rnorm(50 * 5), 50, 5,
            dimnames = list(NULL, paste0("V", 1:5)))
y <- stats::rlnorm(50, meanlog = 0, sdlog = 0.5)

HP <- list(C = 10, eps = 5, rbf_sigma = 1, degree = 2L,
           scale_factor = 1, a = 1L)

K_rbf  <- make_kernel("rbf", sigma = HP$rbf_sigma)
K_poly <- make_kernel("polynomial", degree = HP$degree,
                      coef0 = HP$scale_factor)
K_lin  <- make_kernel("linear")

# Internal accessor for namespace internals.
ns <- asNamespace("psvr")

run_smo <- function(K, sym = NULL) {
  if (is.null(sym)) {
    Omega <- ns$kernel_matrix(K, X)
    diag(Omega) <- diag(Omega) + 1e-6
  } else {
    # Mirror .fit_mape_sym: build Omega_s via sym_kernel_matrix, jitter 0.5e-6,
    # then call adaptive spectral shift (F3, Algorithm 2). Use the same
    # call signature .fit_mape_sym uses.
    Omega <- ns$sym_kernel_matrix(K, X, a = sym)
    diag(Omega) <- diag(Omega) + 0.5e-6
    sh    <- ns$.adaptive_spectral_shift(Omega)
    Omega <- sh$Omega_use
  }
  K_acc <- ns$.make_kernel_accessor(Omega)
  out <- suppressWarnings(
    ns$.smo_solve(K_acc, y, C = HP$C, eps = HP$eps)
  )
  list(iterations = out$iterations, converged = out$converged)
}

cases <- list(
  "mape rbf"      = list(K = K_rbf,  sym = NULL),
  "mape poly"     = list(K = K_poly, sym = NULL),
  "mape lin"      = list(K = K_lin,  sym = NULL),
  "mape sym rbf"  = list(K = K_rbf,  sym = HP$a),
  "mape sym poly" = list(K = K_poly, sym = HP$a),
  "mape sym lin"  = list(K = K_lin,  sym = HP$a)
)

results <- list()
for (nm in names(cases)) {
  cs <- cases[[nm]]
  results[[nm]] <- run_smo(cs$K, cs$sym)
}

cat(sprintf("%-15s | %8s | %s\n", "case", "iters", "converged"))
cat(strrep("-", 40), "\n", sep = "")
for (nm in names(results)) {
  r <- results[[nm]]
  cat(sprintf("%-15s | %8d | %s\n", nm, r$iterations, r$converged))
}

saveRDS(results, out_path)
cat(sprintf("\nSaved to: %s\n", out_path))
