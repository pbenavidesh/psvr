# run_all_experiments.R
# ─────────────────────────────────────────────────────────────
# 30-seed benchmark for psvr cross-section case studies (Phase 3).
# Tidymodels-based pipeline. Run from RStudio with working
# directory set to:
#   vignettes/articles/case-studies/
#
# PARALLELISATION: seeds distributed across workers via furrr.
# Each seed runs 13 models sequentially within its worker.
# Partial results saved per-seed — run is resumable if
# interrupted.
#
# MULTI-MACHINE SETUP:
#   Machine 1: DATASETS_TO_RUN <- c("boston", "diabetes")
#   Machine 2: DATASETS_TO_RUN <- c("energy_efficiency")
#   Copy all result CSVs to one machine before rendering .qmd.
#
# PACKAGES REQUIRED:
#   tidyverse, tidymodels, furrr, future, psvr, mlbench, lars,
#   readxl, kernlab, ranger, xgboost, quantreg, e1071, hardhat
# ─────────────────────────────────────────────────────────────

library(tidyverse)
library(tidymodels)
library(furrr)
library(future)
library(psvr)
library(mlbench)
library(lars)
library(readxl)
library(kernlab)
library(ranger)
library(xgboost)
library(quantreg)
library(e1071)
library(hardhat)

tidymodels_prefer()

library(here)
here::i_am("vignettes/articles/case-studies/run_all_experiments.R")
source(here::here("vignettes/articles/case-studies/experiment_helpers.R"))

# ── CONFIGURATION ─────────────────────────────────────────────
# Edit this vector to control which datasets this machine runs.
DATASETS_TO_RUN <- c("boston", "diabetes", "energy_efficiency")

# DATASETS_TO_RUN <- c("energy_efficiency")

# N_WORKERS <- max(1L, parallel::detectCores() - 1L)
N_WORKERS <- 12
SEEDS <- 1:30

message(sprintf("Workers available: %d", N_WORKERS))
message(sprintf(
  "Datasets to run:   %s\n",
  paste(DATASETS_TO_RUN, collapse = ", ")
))

# ── PARALLEL EXPERIMENT RUNNER ────────────────────────────────
# Parallelises over seeds. Each completed seed is saved as a
# partial RDS so the run is resumable if interrupted.

# ── DATA PREPARATION ──────────────────────────────────────────

datasets <- list()

if ("boston" %in% DATASETS_TO_RUN) {
  data("BostonHousing", package = "mlbench")
  df_b <- BostonHousing |>
    dplyr::mutate(chas = as.integer(chas))
  datasets$boston <- list(
    X = df_b |> dplyr::select(-medv) |> as.matrix(),
    y = df_b$medv,
    name = "boston"
  )
  stopifnot(all(datasets$boston$y > 0))
  message("Boston Housing loaded: n =", nrow(datasets$boston$X))
}

if ("diabetes" %in% DATASETS_TO_RUN) {
  data("diabetes", package = "lars")
  X_diab_clean <- matrix(
    as.numeric(diabetes$x),
    nrow = nrow(diabetes$x),
    ncol = ncol(diabetes$x),
    dimnames = list(NULL, colnames(diabetes$x))
  )
  datasets$diabetes <- list(
    X = X_diab_clean,
    y = as.numeric(diabetes$y),
    name = "diabetes"
  )
  stopifnot(all(datasets$diabetes$y > 0))
  message("Diabetes loaded: n =", nrow(datasets$diabetes$X))
}

if ("energy_efficiency" %in% DATASETS_TO_RUN) {
  ee_url <- paste0(
    "https://archive.ics.uci.edu/ml/machine-learning-databases/",
    "00242/ENB2012_data.xlsx"
  )
  ee_local <- here::here(
    "vignettes/articles/case-studies/results",
    "ENB2012_data.xlsx"
  )
  if (!file.exists(ee_local)) {
    message("Downloading Energy Efficiency dataset...")
    download.file(ee_url, ee_local, mode = "wb", quiet = TRUE)
  }
  df_ee <- readxl::read_excel(ee_local)
  datasets$energy_efficiency <- list(
    X = df_ee |> dplyr::select(X1:X8) |> as.matrix(),
    y = df_ee$Y1,
    name = "energy_efficiency"
  )
  stopifnot(all(datasets$energy_efficiency$y > 0))
  message("Energy Efficiency loaded: n =", nrow(datasets$energy_efficiency$X))
}

# ── RUN ───────────────────────────────────────────────────────

for (ds in datasets) {
  out_csv <- here::here(
    "vignettes/articles/case-studies/results",
    sprintf("%s-results.csv", gsub("_", "-", ds$name))
  )
  if (file.exists(out_csv)) {
    message(sprintf(
      "\n[%s] Final CSV exists — skipping. Delete to re-run.\n",
      ds$name
    ))
    next
  }
  run_experiment_parallel(
    ds$X,
    ds$y,
    ds$name,
    seeds = SEEDS,
    workers = N_WORKERS
  )
}

# ── FINAL VERIFICATION ────────────────────────────────────────

message("\n── Verification ──────────────────────────────────────")
for (ds_name in DATASETS_TO_RUN) {
  f <- here::here(
    "vignettes/articles/case-studies/results",
    sprintf("%s-results.csv", gsub("_", "-", ds_name))
  )
  if (file.exists(f)) {
    df <- readr::read_csv(f, show_col_types = FALSE)
    nas <- sum(is.na(df$MAPE))
    message(sprintf(
      "  %-42s rows: %4d  NA(MAPE): %d",
      basename(f),
      nrow(df),
      nas
    ))
  } else {
    message(sprintf("  %-42s NOT YET RUN", basename(f)))
  }
}
message("──────────────────────────────────────────────────────\n")
