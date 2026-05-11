devtools::load_all("C:/Users/behep/OneDrive - ITESO/PhD/00-Tesis/psvr",
                   recompile = TRUE, quiet = TRUE)

set.seed(2026)
N <- 50
X <- matrix(rnorm(N * 3), N, 3)
y <- rlnorm(N, sdlog = 0.5)

K <- make_kernel("rbf", sigma = 1)

# Cold path (no precompute)
fit_cold <- psvr(X, y, loss = "mape", kernel = K, C = 10, eps = 5)

# Precompute Omega and pass it in
Omega_pre <- psvr:::kernel_matrix(K, X)
fit_pre   <- psvr(X, y, loss = "mape", kernel = K, C = 10, eps = 5,
                  precomputed_Omega = Omega_pre)

cat("alpha     identical: ", identical(fit_cold$alpha, fit_pre$alpha), "\n")
cat("alpha*    identical: ", identical(fit_cold$alpha_star, fit_pre$alpha_star), "\n")
cat("beta      identical: ", identical(fit_cold$beta, fit_pre$beta), "\n")
cat("b         identical: ", identical(fit_cold$b, fit_pre$b), "\n")
cat("iters     match:     ", fit_cold$solver_meta$iters, "==", fit_pre$solver_meta$iters,
    "->", identical(fit_cold$solver_meta$iters, fit_pre$solver_meta$iters), "\n")
cat("converged match:     ", fit_cold$solver_meta$converged, "==", fit_pre$solver_meta$converged, "\n")

# Symmetric path
fit_sym_cold <- psvr(X, y, loss = "mape", sym = +1L, kernel = K, C = 10, eps = 5)
Omega_s_pre  <- psvr:::sym_kernel_matrix(K, X, a = 1L)
fit_sym_pre  <- psvr(X, y, loss = "mape", sym = +1L, kernel = K, C = 10, eps = 5,
                     precomputed_Omega_s = Omega_s_pre)

cat("\nSymmetric (a = +1):\n")
cat("alpha     identical: ", identical(fit_sym_cold$alpha, fit_sym_pre$alpha), "\n")
cat("beta      identical: ", identical(fit_sym_cold$beta, fit_sym_pre$beta), "\n")
cat("b         identical: ", identical(fit_sym_cold$b, fit_sym_pre$b), "\n")
cat("iters     match:     ", fit_sym_cold$solver_meta$iters, "==", fit_sym_pre$solver_meta$iters, "\n")

# Asymmetric subset (mimic the per-fold slice psvr_cv() will use)
train_idx <- 1:40   # 40-row "fold training set"
Omega_full <- psvr:::kernel_matrix(K, X)
Omega_sub  <- Omega_full[train_idx, train_idx]
fit_sub_pre  <- psvr(X[train_idx, ], y[train_idx], loss = "mape",
                     kernel = K, C = 10, eps = 5,
                     precomputed_Omega = Omega_sub)
fit_sub_cold <- psvr(X[train_idx, ], y[train_idx], loss = "mape",
                     kernel = K, C = 10, eps = 5)
cat("\nSubset (train_idx = 1:40):\n")
cat("alpha     identical: ", identical(fit_sub_cold$alpha, fit_sub_pre$alpha), "\n")
cat("b         identical: ", identical(fit_sub_cold$b, fit_sub_pre$b), "\n")
