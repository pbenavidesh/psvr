# Fit a least-squares SVR with RMSPE loss

Fits the percentage-error LS-SVR of the paper: Model 3 when
`sym_type = "none"`, and the symmetric-kernel Model 4 when `sym_type` is
`"even"` or `"odd"`. There is no quadratic program and no sparsity: the
fit is a single solve of the \\(N+1) \times (N+1)\\ augmented linear
system \$\$\begin{pmatrix} 0 & 1^{\top} \\ 1 & \Omega +
Y\_{\Gamma}\end{pmatrix} \begin{pmatrix} b \\ \alpha \end{pmatrix} =
\begin{pmatrix} 0 \\ y \end{pmatrix}\$\$ with \\Y\_{\Gamma} =
\mathrm{diag}(y_1^2/\Gamma, \ldots, y_N^2/\Gamma)\\, and \\\Omega_s\\
replacing \\\Omega\\ in the symmetric case. Every training point
contributes to the prediction.

## Usage

``` r
psvr_rmspe(
  X,
  y,
  sym_type = c("none", "even", "odd"),
  kernel,
  gamma,
  precondition = "auto",
  ...
)
```

## Arguments

- X:

  Numeric matrix of training inputs, one observation per row (\\N \times
  p\\).

- y:

  Numeric vector of training targets, length \\N\\. Must satisfy \\y_k
  \> 0\\ for every \\k\\; percentage-error loss is undefined otherwise,
  and this is checked rather than coerced.

- sym_type:

  Symmetry type, one of `"none"` (default), `"even"` or `"odd"`. Maps
  onto the symmetry parameter \\a\\ of the paper: `"none"` fits Model 3
  and imposes no symmetry constraint; `"even"` sets \\a = +1\\,
  enforcing \\f(x) = f(-x)\\; `"odd"` sets \\a = -1\\, enforcing \\f(x)
  = -f(-x)\\. This is the same vocabulary as the `sym_type` argument of
  the parsnip specifications, so the two public surfaces agree. The
  symmetric variants require a kernel satisfying Assumption 3 of the
  paper – see
  [`make_kernel()`](https://pbenavidesh.github.io/psvr/reference/make_kernel.md).

- kernel:

  A kernel function created by
  [`make_kernel()`](https://pbenavidesh.github.io/psvr/reference/make_kernel.md).

- gamma:

  Regularization parameter \\\Gamma \> 0\\. Required. Larger values
  weight the squared percentage residuals more heavily against the norm
  penalty.

- precondition:

  One of `"auto"` (default), `"always"`, `"never"`, or a positive
  numeric threshold. Controls the symmetric rescaling of Remark 17.
  `"auto"` applies it when the target ratio \\\max(y)/\min(y)\\ exceeds
  10; a numeric value sets that threshold explicitly. Whether it fired
  is reported in `fit$precondition_applied`.

- ...:

  Must be empty. Passing anything here is an error, which is how a
  mistyped argument name is caught.

## Value

For `sym_type = "none"`, an object of class `"psvr_rmspe"`: a list with
components `alpha` (the length-\\N\\ multipliers), `b`, `X_train`,
`y_train`, `fitted_values`, `kernel`, `gamma`, `n_train`, `p_train` and
`precondition_applied`.

For `sym_type = "even"` or `"odd"`, an object of class
`"psvr_rmspe_sym"`: the same components plus `a` (the symmetry
parameter).

Methods are available for
[`predict()`](https://rdrr.io/r/stats/predict.html),
[`print()`](https://rdrr.io/r/base/print.html),
[`coef()`](https://rdrr.io/r/stats/coef.html),
[`summary()`](https://rdrr.io/r/base/summary.html),
[`fitted()`](https://rdrr.io/r/stats/fitted.values.html) and
[`residuals()`](https://rdrr.io/r/stats/residuals.html). Note that
[`coef()`](https://rdrr.io/r/stats/coef.html) returns three components
here (`alpha`, `b`, `support_data`) against five for the MAPE classes:
LS-SVR has no `alpha_star` and no pruned `beta`, and the absent
components are not materialised as `NULL`.

## Details

For the epsilon-SVR / MAPE family (Models 1 and 2) see
[`psvr_mape()`](https://pbenavidesh.github.io/psvr/reference/psvr_mape.md).
The two are deliberately separate functions: they share no solver, no
dual structure and no hyperparameter search space. The name
[`psvr()`](https://pbenavidesh.github.io/psvr/reference/psvr-package.md)
is reserved for a future automatic-selection front end and is **not** a
synonym for either.

## See also

[`psvr_mape()`](https://pbenavidesh.github.io/psvr/reference/psvr_mape.md)
for the epsilon-SVR / MAPE family,
[`make_kernel()`](https://pbenavidesh.github.io/psvr/reference/make_kernel.md)
for kernels.

## Examples

``` r
set.seed(1)
X <- matrix(rnorm(40), 20, 2)
y <- rlnorm(20)
K <- make_kernel("rbf", sigma = 1)

fit <- psvr_rmspe(X, y, kernel = K, gamma = 100)
predict(fit, X[1:3, , drop = FALSE])
#> [1] 0.9189556 0.7369731 0.8524386

# Even-symmetric variant (Model 4): f(x) = f(-x).
fit_sym <- psvr_rmspe(X, y, sym_type = "even", kernel = K, gamma = 100)
predict(fit_sym, X[1:3, , drop = FALSE])
#> [1] 1.1583805 0.7439030 0.5084087
```
