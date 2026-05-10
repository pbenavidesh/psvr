## Run the full testthat suite. Used during F1 verification.
Sys.setenv(NOT_CRAN = "true")
suppressPackageStartupMessages(devtools::load_all(".", quiet = TRUE))
testthat::test_dir(
  "tests/testthat",
  reporter = testthat::SummaryReporter$new(),
  stop_on_failure = FALSE
)
