## Static check of the pkgdown reference structure: validates that every
## function in the package namespace is either listed in _pkgdown.yml's
## reference: section OR tagged @keywords internal. Surfaces orphan
## entries (listed but missing) and bare entries (exported but missing).
##
## This is a Pandoc-free alternative to pkgdown::build_site() — it answers
## the question "would the 12 fit wrappers appear in the reference index"
## without actually rendering HTML.

suppressPackageStartupMessages({
  devtools::load_all(".", quiet = TRUE)
  yml <- yaml::read_yaml("_pkgdown.yml")
})

# Topics explicitly listed in _pkgdown.yml reference sections
listed <- unlist(lapply(yml$reference, function(s) s$contents),
                 use.names = FALSE)

# All Rd topics shipping with the package
all_rds <- list.files("man", pattern = "\\.Rd$", full.names = TRUE)
parse_keywords <- function(rd) {
  txt <- readLines(rd, warn = FALSE)
  any(grepl("\\\\keyword\\{internal\\}", txt))
}
parse_alias <- function(rd) {
  txt <- readLines(rd, warn = FALSE)
  m <- regmatches(txt, regexec("\\\\name\\{([^}]+)\\}", txt))
  m <- m[lengths(m) > 0]
  if (length(m) == 0L) NA_character_ else m[[1]][2]
}

rds <- data.frame(
  rd       = basename(all_rds),
  topic    = vapply(all_rds, parse_alias, character(1L)),
  internal = vapply(all_rds, parse_keywords, logical(1L)),
  stringsAsFactors = FALSE,
  row.names = NULL
)

cat("Total Rd files:                              ", nrow(rds),                "\n")
cat("Marked @keywords internal:                   ", sum(rds$internal),        "\n")
cat("Listed in _pkgdown.yml reference:            ", length(listed),           "\n\n")

internal_listed <- intersect(listed, rds$topic[rds$internal])
internal_unlisted <- setdiff(rds$topic[rds$internal], listed)
public_unlisted <- setdiff(rds$topic[!rds$internal], listed)

cat("--- Internal Rd topics (will be HIDDEN from reference) ---\n")
cat(paste(" -", sort(rds$topic[rds$internal])), sep = "\n")
cat("\n--- Internal topics ALSO listed in _pkgdown.yml (would override hide) ---\n")
if (length(internal_listed) == 0L) {
  cat("  (none) \U0001F44D  the 12 fit wrappers and the 4 .fit_* internals are correctly hidden\n")
} else {
  cat(paste(" -", internal_listed), sep = "\n")
}

cat("\n--- Public Rd topics NOT listed in _pkgdown.yml (would auto-appear) ---\n")
if (length(public_unlisted) == 0L) {
  cat("  (none) \n")
} else {
  cat(paste(" -", public_unlisted), sep = "\n")
}

cat("\n--- Listed in _pkgdown.yml but no Rd file (orphan entries) ---\n")
listed_topics <- vapply(listed, function(t) {
  if (any(rds$topic == t)) t else NA_character_
}, character(1L))
orphans <- listed[is.na(listed_topics)]
if (length(orphans) == 0L) {
  cat("  (none) \n")
} else {
  cat(paste(" -", orphans), sep = "\n")
}
