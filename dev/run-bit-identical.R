## Helper script: run the bit-identical golden tests once.
## Used during F1 Step 1 to generate / verify snapshots.
Sys.setenv(NOT_CRAN = "true")            # opt expect_snapshot_value() in
suppressPackageStartupMessages(devtools::load_all(".", quiet = TRUE))
testthat::test_file(
  "tests/testthat/test-bit-identical.R",
  reporter = testthat::SummaryReporter$new()
)
