## Regenerate NAMESPACE and Rd files from roxygen comments.
suppressPackageStartupMessages(devtools::document(roclets = c("rd", "namespace")))
