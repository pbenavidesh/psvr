## Build only the pkgdown reference (skips vignettes/articles which need
## Pandoc). Output goes to ./docs/reference/.
suppressPackageStartupMessages(devtools::load_all(".", quiet = TRUE))
unlink("docs/reference", recursive = TRUE, force = TRUE)
pkgdown::init_site()
pkgdown::build_reference(lazy = FALSE, examples = FALSE)
cat("\n----- docs/reference contents -----\n")
files <- list.files("docs/reference", pattern = "\\.html$")
cat(paste(" -", sort(files)), sep = "\n")
