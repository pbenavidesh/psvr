## Compare F3 vs F4 predictions and emit a drift+iter table.
##
## Reads dev/preds_F3.rds, dev/preds_F4.rds, dev/iters_F3.rds, dev/iters_F4.rds
## and prints one row per snapshot test with the following columns:
##   F3 iter | F3 conv | F4 iter | F4 conv | Max abs diff | Verdict
##
## Verdict categorisation:
##   F4 FIX    -> F3 stalled (max_iter, !converged) but F4 converged.
##   STALE F3+F4 -> Both stalled at max_iter; "drift" is between two
##                  non-converged states (neither is correct).
##   LINSOLVE  -> RMSPE test; LS-SVR linear system, not SMO. Always 0 drift.
##   LEGIT     -> Both converged; drift within ~tol*max(y).
##   BUG?      -> Both converged but drift >> 10 * tol*max(y).

f3 <- readRDS("dev/preds_F3.rds")
f4 <- readRDS("dev/preds_F4.rds")
iters_f3 <- readRDS("dev/iters_F3.rds")
iters_f4 <- readRDS("dev/iters_F4.rds")

stopifnot(identical(names(f3), names(f4)))

# Snapshot fixture max(y) for the "expected drift" scaling.
set.seed(2026)
y_fix <- stats::rlnorm(50, meanlog = 0, sdlog = 0.5)
y_max <- max(y_fix)
tol   <- 1e-3
expected_per_pred <- tol * y_max

cat(sprintf("Snapshot fixture: max(y) = %.4f, tol = %.0e\n", y_max, tol))
cat(sprintf("Expected per-pair drift order: ~%.2e\n", expected_per_pred))
cat(sprintf("Red-flag threshold (10x):       %.2e\n\n", 10 * expected_per_pred))

# Map a test name to a SMO case key in iters_F3/F4 (or NULL for RMSPE).
classify_test <- function(name) {
  if (grepl("rmspe", name, ignore.case = TRUE)) return(NULL)
  # Sym tests are tagged by "_sym_", "sym=+1", or "Model 2" (Model 1 = no-sym,
  # Model 2 = sym). All other MAPE tests are no-sym.
  has_sym <- grepl("_sym_|sym=\\+1|Model 2", name)
  is_rbf  <- grepl("rbf|RBF", name)
  is_poly <- grepl("poly", name)
  is_lin  <- grepl("linear", name)
  if (has_sym) {
    if (is_rbf)  return("mape sym rbf")
    if (is_poly) return("mape sym poly")
    if (is_lin)  return("mape sym lin")
  } else {
    if (is_rbf)  return("mape rbf")
    if (is_poly) return("mape poly")
    if (is_lin)  return("mape lin")
  }
  NULL
}

to_num <- function(x) {
  if (is.list(x) && "predictions" %in% names(x)) return(as.numeric(x$predictions))
  if (is.data.frame(x)) return(as.numeric(x[[1]]))
  as.numeric(x)
}

verdict <- function(f3_conv, f4_conv, max_diff, smo_case) {
  if (is.null(smo_case)) return("LINSOLVE")
  if (!f3_conv && !f4_conv) return("STALE F3+F4")
  if (!f3_conv &&  f4_conv) return("F4 FIX")
  if ( f3_conv && !f4_conv) return("F4 REGRESS")
  # both converged
  if (max_diff > 10 * expected_per_pred) return("BUG?")
  "LEGIT"
}

rows <- vector("list", length(f3))
for (i in seq_along(f3)) {
  v3 <- to_num(f3[[i]])
  v4 <- to_num(f4[[i]])
  d  <- abs(v4 - v3)
  smo_case <- classify_test(names(f3)[[i]])
  if (is.null(smo_case)) {
    f3_iter <- NA_integer_; f3_conv <- NA
    f4_iter <- NA_integer_; f4_conv <- NA
  } else {
    f3_iter <- iters_f3[[smo_case]]$iterations
    f3_conv <- iters_f3[[smo_case]]$converged
    f4_iter <- iters_f4[[smo_case]]$iterations
    f4_conv <- iters_f4[[smo_case]]$converged
  }
  rows[[i]] <- data.frame(
    test     = names(f3)[[i]],
    f3_iter  = f3_iter, f3_conv = f3_conv,
    f4_iter  = f4_iter, f4_conv = f4_conv,
    max_diff = max(d),
    verdict  = verdict(f3_conv, f4_conv, max(d), smo_case),
    stringsAsFactors = FALSE
  )
}
df <- do.call(rbind, rows)
df <- df[order(factor(df$verdict,
                      levels = c("BUG?", "F4 REGRESS", "STALE F3+F4",
                                 "F4 FIX", "LEGIT", "LINSOLVE")),
              -df$max_diff), ]

fmt_iter <- function(x) if (is.na(x)) "  -   " else sprintf("%6d", x)
fmt_conv <- function(x) if (is.na(x)) "  -  " else if (x) " YES " else " NO  "

cat("| Test name | F3 iter | F3 conv | F4 iter | F4 conv | Max abs diff | Verdict |\n")
cat("|---|---|---|---|---|---|---|\n")
for (i in seq_len(nrow(df))) {
  cat(sprintf("| %s | %s | %s | %s | %s | %.3e | %s |\n",
              df$test[i],
              fmt_iter(df$f3_iter[i]),
              fmt_conv(df$f3_conv[i]),
              fmt_iter(df$f4_iter[i]),
              fmt_conv(df$f4_conv[i]),
              df$max_diff[i],
              df$verdict[i]))
}

cat("\nVerdict counts:\n")
print(table(df$verdict))
cat("\nKey numbers:\n")
cat(sprintf("  Total tests:       %d\n", nrow(df)))
cat(sprintf("  BUG? (red flag):   %d\n", sum(df$verdict == "BUG?")))
cat(sprintf("  Max drift overall: %.3e\n", max(df$max_diff)))

# Iteration speedup summary for the cases that converged in both.
ok <- !is.na(df$f3_iter) & df$f3_conv & df$f4_conv
if (any(ok)) {
  cat("\nIter-count comparison (both converged):\n")
  for (i in which(ok)) {
    cat(sprintf("  %-55s F3: %5d  F4: %5d  (%+d, %.1f%%)\n",
                df$test[i], df$f3_iter[i], df$f4_iter[i],
                df$f4_iter[i] - df$f3_iter[i],
                100 * (df$f4_iter[i] - df$f3_iter[i]) / df$f3_iter[i]))
  }
}
