devtools::load_all("C:/Users/behep/OneDrive - ITESO/PhD/00-Tesis/psvr", quiet = TRUE)

set.seed(42)
X <- matrix(rnorm(50), 10, 5)

K_rbf  <- make_kernel("rbf", sigma = 1)
K_lin  <- make_kernel("linear")
K_poly <- make_kernel("polynomial", degree = 3, coef0 = 1)

# R reference (existing nested-loop kernel_matrix — Step 3 will modify it,
# but at this commit it is still the original closure-loop).
M_r_rbf  <- psvr:::kernel_matrix(K_rbf,  X)
M_r_lin  <- psvr:::kernel_matrix(K_lin,  X)
M_r_poly <- psvr:::kernel_matrix(K_poly, X)

# Direct C++
M_c_rbf  <- psvr:::kernel_rbf_cpp(X, X, 1)
M_c_lin  <- psvr:::kernel_linear_cpp(X, X)
M_c_poly <- psvr:::kernel_poly_cpp(X, X, 1, 3L)

cat("RBF        max|R - cpp| =", format(max(abs(M_r_rbf  - M_c_rbf)),  digits = 17), "\n")
cat("linear     max|R - cpp| =", format(max(abs(M_r_lin  - M_c_lin)),  digits = 17), "\n")
cat("polynomial max|R - cpp| =", format(max(abs(M_r_poly - M_c_poly)), digits = 17), "\n")

cat("RBF        identical: ", identical(M_r_rbf,  M_c_rbf),  "\n")
cat("linear     identical: ", identical(M_r_lin,  M_c_lin),  "\n")
cat("polynomial identical: ", identical(M_r_poly, M_c_poly), "\n")

# Asymmetric case (X vs -X)
M_r_rbf_neg <- psvr:::kernel_matrix(K_rbf, X, -X)
M_c_rbf_neg <- psvr:::kernel_rbf_cpp(X, -X, 1)
cat("RBF        K(X, -X) identical: ", identical(M_r_rbf_neg, M_c_rbf_neg), "\n")
