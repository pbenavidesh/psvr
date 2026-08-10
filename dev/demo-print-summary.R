## Capture print(fit) and summary(fit) for the user-specified demo.
suppressPackageStartupMessages({
  library(devtools)
  load_all()
})

set.seed(2026)
X <- matrix(rnorm(50 * 5), 50, 5)
y <- rlnorm(50)

fit <- psvr_mape(X, y,
                 kernel = make_kernel("rbf", sigma = 1),
                 C      = 10,
                 eps    = 5)

print(fit)
cat("\n---\n")
summary(fit)
