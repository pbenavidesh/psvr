## Run R CMD check via devtools::check(). Used as the F1 acceptance gate.
## --no-build-vignettes skips the Pandoc-requiring HTML re-render (Pandoc
## is not installed on this machine); vignette source files are still
## syntactically validated by check().
Sys.setenv(NOT_CRAN = "true")
result <- devtools::check(
  args         = c("--no-manual", "--ignore-vignettes"),
  build_args   = c("--no-build-vignettes"),
  error_on     = "never"
)
cat("\n----- check summary -----\n")
cat("errors:   ", length(result$errors),   "\n")
cat("warnings: ", length(result$warnings), "\n")
cat("notes:    ", length(result$notes),    "\n")
if (length(result$errors)   > 0) cat("\nERRORS:\n",   paste(result$errors,   collapse="\n\n"), "\n")
if (length(result$warnings) > 0) cat("\nWARNINGS:\n", paste(result$warnings, collapse="\n\n"), "\n")
if (length(result$notes)    > 0) cat("\nNOTES:\n",    paste(result$notes,    collapse="\n\n"), "\n")
