# ============================================================================
#  Synthetic parity-test ablation — 2x2 design on data with TRUE even symmetry
# ============================================================================
#  Purpose
#    Demonstrate that the symmetric-kernel modification (Theorems 2 and 4) AND
#    the equivalent reflection augmentation IMPROVE over the non-symmetric
#    baseline when the target function is genuinely even-symmetric — providing
#    the empirical validation promised in the original Wave 1 commitment to
#    Reviewer 1.
#
#  Design
#    y_k = 5 + ||x_k||^2 + epsilon_k,   x_k ~ N(0, I_p),   epsilon_k ~ N(0, sigma_eps^2)
#    By construction f(x) = f(-x), so the a = +1 even-symmetry assumption
#    holds exactly. We expect:
#      - Family A: A2 ≈ A4 < A1, A3 < A1 (kernel ≡ aug, both improve baseline)
#      - Family B: B2 ≈ B4 < B1, B3 < B1
#    On Boston the same design produced the OPPOSITE pattern; the contrast
#    isolates the effect of the symmetry assumption holding vs not holding.
# ============================================================================

library(tidyverse)
library(tidymodels)
library(psvr)
library(furrr)
library(future)
library(here)

source(here::here("vignettes/articles/case-studies/experiment_helpers.R"))
source(here::here("vignettes/articles/case-studies/boston_ablation_helpers.R"))

# -- Configuration ------------------------------------------------------------

N_SEEDS <- 30L
N_WORKERS <- max(1L, parallel::detectCores() - 1L)

N_SAMPLES <- 400L # total samples (300 train / 100 test typical at 75/25)
P_FEATURES <- 5L
SIGMA_EPS <- 0.5 # noise sd; small enough to keep all y > 0
DATA_SEED <- 20260429L

results_file <- here::here(
  "vignettes/articles/case-studies/results",
  "synthetic-parity-ablation-results.csv"
)

# -- Synthetic data generation ------------------------------------------------
# Single fixed dataset; the 30 seeds vary the train/test split, mirroring the
# Boston ablation protocol so the two experiments are directly comparable.

set.seed(DATA_SEED)

X <- matrix(rnorm(N_SAMPLES * P_FEATURES), nrow = N_SAMPLES, ncol = P_FEATURES)
colnames(X) <- paste0("x", seq_len(P_FEATURES))

y_true <- 5 + rowSums(X^2) # exactly even: f(-x) = f(x)
y <- y_true + rnorm(N_SAMPLES, sd = SIGMA_EPS)

stopifnot(all(y > 0)) # MAPE requires strict positivity

cat(sprintf(
  "Synthetic data: N = %d, p = %d, range(y) = [%.2f, %.2f], mean(y) = %.2f\n",
  N_SAMPLES,
  P_FEATURES,
  min(y),
  max(y),
  mean(y)
))

# Sanity: check that X has zero-mean (symmetry of inputs is assumed by the
# data-augmentation pipeline; we trust the random draw but verify).
cat("Mean of each column of X (should be ~0):\n")
print(round(colMeans(X), 3))

# -- Run experiment -----------------------------------------------------------

if (!file.exists(results_file)) {
  results <- run_experiment_ablation_parallel(
    X,
    y,
    seeds = seq_len(N_SEEDS),
    workers = N_WORKERS,
    dataset_name = "synthetic_parity"
  )
} else {
  results <- read_csv(results_file, show_col_types = FALSE)
}

cat("Rows loaded:", nrow(results), "  (expected:", 8L * N_SEEDS, ")\n")
cat("NA-MAPE rows:", sum(is.na(results$MAPE)), "\n")

# -- Quick analysis -----------------------------------------------------------

boot_ci_mean <- function(x, B = 1000L, alpha = 0.05) {
  x <- x[is.finite(x)]
  if (length(x) < 2L) {
    return(c(NA_real_, NA_real_))
  }
  m <- replicate(B, mean(sample(x, length(x), replace = TRUE)))
  unname(quantile(m, c(alpha / 2, 1 - alpha / 2)))
}

cell_summary <- results |>
  group_by(family, cell_id, sym, augmented) |>
  summarise(
    mean_MAPE = mean(MAPE, na.rm = TRUE),
    sd_MAPE = sd(MAPE, na.rm = TRUE),
    ci_lo = boot_ci_mean(MAPE)[1],
    ci_hi = boot_ci_mean(MAPE)[2],
    mean_R2 = mean(R2, na.rm = TRUE),
    n = sum(!is.na(MAPE)),
    .groups = "drop"
  ) |>
  arrange(family, cell_id)

cat("\n=== Per-cell summary (synthetic parity) ===\n")
print(cell_summary, n = 8)

# Wilcoxon vs baseline (cell 1) per family
wilcoxon_vs_baseline <- function(res, fam_letter) {
  fam <- res |>
    filter(family == fam_letter) |>
    select(seed, cell_id, MAPE) |>
    pivot_wider(names_from = cell_id, values_from = MAPE)

  baseline <- if (fam_letter == "A") "A1" else "B1"
  others <- if (fam_letter == "A") c("A2", "A3", "A4") else c("B2", "B3", "B4")

  do.call(
    rbind,
    lapply(others, function(c2) {
      diff <- fam[[c2]] - fam[[baseline]]
      p <- tryCatch(
        wilcox.test(diff, alternative = "less", paired = FALSE)$p.value,
        error = function(e) NA_real_
      )
      tibble(
        family = fam_letter,
        comparison = paste(c2, "vs", baseline),
        median_diff = median(diff, na.rm = TRUE),
        p_value = p,
        direction = if (median(diff, na.rm = TRUE) < 0) "improves" else "worse"
      )
    })
  )
}

wilcoxon_results <- bind_rows(
  wilcoxon_vs_baseline(results, "A"),
  wilcoxon_vs_baseline(results, "B")
)

cat("\n=== Paired Wilcoxon vs baseline (one-sided 'less', i.e. improves) ===\n")
print(wilcoxon_results)

cat(
  "\nExpected pattern on parity data: median_diff < 0 and p < 0.05 in all 6 rows.\n"
)
cat(
  "If all 6 rows show 'improves' significantly, the symmetric extensions are validated.\n"
)
