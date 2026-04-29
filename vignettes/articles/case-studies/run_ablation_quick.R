# Standalone ablation runner
# Corre desde la raíz del proyecto con:
#   Rscript vignettes/articles/case-studies/run_ablation_quick.R

suppressPackageStartupMessages({
  library(here)
  library(dplyr)
  library(tidymodels)
  library(psvr)
  library(mlbench)
  library(readr)
})

source(here::here("vignettes/articles/case-studies/experiment_helpers.R"))
source(here::here("vignettes/articles/case-studies/boston_ablation_helpers.R"))

# ── Configuration ────────────────────────────────────────────────
N_SEEDS    <- 30L
N_WORKERS  <- 21L
EXCLUDE_B  <- FALSE
LHS_BUDGET <- 50L

# ── Data ─────────────────────────────────────────────────────────
data("BostonHousing", package = "mlbench")
df <- BostonHousing |> mutate(chas = as.integer(chas))
if (EXCLUDE_B) df <- df |> select(-b)
y  <- df$medv
X  <- df |> select(-medv) |> as.matrix()
stopifnot(all(y > 0))

cat("Boston: n =", nrow(X), "  p =", ncol(X), "\n")
cat("Seeds:", N_SEEDS, "  Workers:", N_WORKERS, "\n\n")

# ── Run ──────────────────────────────────────────────────────────
results <- run_experiment_ablation_parallel(
  X, y,
  seeds        = seq_len(N_SEEDS),
  workers      = N_WORKERS,
  dataset_name = "boston_ablation"
)

# ── Save ─────────────────────────────────────────────────────────
results_file <- here::here(
  "vignettes/articles/case-studies/results",
  "boston-reflection-ablation-results.csv"
)
readr::write_csv(results, results_file)

cat("Done. Rows:", nrow(results),
    " NA-MAPE:", sum(is.na(results$MAPE)), "\n")
cat("Wrote:", results_file, "\n")