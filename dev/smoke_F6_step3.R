devtools::load_all("C:/Users/behep/OneDrive - ITESO/PhD/00-Tesis/psvr",
                   recompile = TRUE, quiet = TRUE)

set.seed(42)
X <- matrix(rnorm(50), 10, 5)

K_rbf  <- make_kernel("rbf", sigma = 1)
K_lin  <- make_kernel("linear")
K_poly <- make_kernel("polynomial", degree = 3, coef0 = 1)

# Dispatch path (kernel_matrix() now routes to Rcpp via kernel_info attr)
M_disp_rbf  <- psvr:::kernel_matrix(K_rbf,  X)
M_disp_lin  <- psvr:::kernel_matrix(K_lin,  X)
M_disp_poly <- psvr:::kernel_matrix(K_poly, X)

# Direct C++ for parity check
M_cpp_rbf  <- psvr:::kernel_rbf_cpp(X, X, 1)
M_cpp_lin  <- psvr:::kernel_linear_cpp(X, X)
M_cpp_poly <- psvr:::kernel_poly_cpp(X, X, 1, 3L)

cat("Dispatched RBF        == direct cpp: ", identical(M_disp_rbf,  M_cpp_rbf),  "\n")
cat("Dispatched linear     == direct cpp: ", identical(M_disp_lin,  M_cpp_lin),  "\n")
cat("Dispatched polynomial == direct cpp: ", identical(M_disp_poly, M_cpp_poly), "\n")

# Legacy R-only path for parity check
M_leg_rbf  <- psvr:::.legacy_kernel_matrix(K_rbf,  X, X)
M_leg_lin  <- psvr:::.legacy_kernel_matrix(K_lin,  X, X)
M_leg_poly <- psvr:::.legacy_kernel_matrix(K_poly, X, X)

cat("Dispatched RBF        == legacy R:   ", identical(M_disp_rbf,  M_leg_rbf),  "\n")
cat("Dispatched linear     == legacy R:   ", identical(M_disp_lin,  M_leg_lin),  "\n")
cat("Dispatched polynomial == legacy R:   ", identical(M_disp_poly, M_leg_poly), "\n")

# User-defined closure (no kernel_info attribute) -> falls through to legacy
K_custom <- function(xi, xj) sum(xi * xj)^2  # custom quadratic
cat("User-defined closure has kernel_info? ",
    !is.null(attr(K_custom, "kernel_info")), "\n")

M_custom_disp <- psvr:::kernel_matrix(K_custom, X, X)
M_custom_leg  <- psvr:::.legacy_kernel_matrix(K_custom, X, X)
cat("Custom closure dispatch == legacy:    ", identical(M_custom_disp, M_custom_leg), "\n")

# Sym path (sym_kernel_matrix calls kernel_matrix twice, including K(X, -X))
M_sym_disp <- psvr:::sym_kernel_matrix(K_rbf, X, a = 1L)
# Reference: compute via the legacy path
Omega_leg     <- psvr:::.legacy_kernel_matrix(K_rbf, X, X)
Omega_neg_leg <- psvr:::.legacy_kernel_matrix(K_rbf, X, -X)
M_sym_leg     <- 0.5 * (Omega_leg + 1L * Omega_neg_leg)
cat("sym_kernel_matrix (a=+1)  identical:  ", identical(M_sym_disp, M_sym_leg), "\n")

M_sym_disp_m1 <- psvr:::sym_kernel_matrix(K_rbf, X, a = -1L)
M_sym_leg_m1  <- 0.5 * (Omega_leg + (-1L) * Omega_neg_leg)
cat("sym_kernel_matrix (a=-1)  identical:  ", identical(M_sym_disp_m1, M_sym_leg_m1), "\n")

# Predict-shaped call (M=1 column)
x_new <- matrix(rnorm(5), nrow = 1, ncol = 5)
M_pred_disp <- psvr:::kernel_matrix(K_rbf, X, x_new)
M_pred_leg  <- psvr:::.legacy_kernel_matrix(K_rbf, X, x_new)
cat("Predict-shape (M=1)       identical:  ", identical(M_pred_disp, M_pred_leg), "\n")
cat("Predict-shape dim:        ", dim(M_pred_disp), "\n")
