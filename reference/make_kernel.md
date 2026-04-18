# Create a kernel function

Returns a closure `K(xi, xj)` suitable for use with all four psvr model
fitting functions. The returned function accepts numeric vectors of any
sign, which is required by the symmetric models (Models 2 and 4) that
evaluate `K(xk, -xl)`.

## Usage

``` r
make_kernel(
  type = c("rbf", "linear", "polynomial"),
  sigma = 1,
  degree = 3L,
  coef0 = 1
)
```

## Arguments

- type:

  Kernel type: `"rbf"`, `"linear"`, or `"polynomial"`.

- sigma:

  Bandwidth for the RBF kernel, `sigma > 0` (default `1`).

- degree:

  Integer degree for the polynomial kernel, `degree >= 1` (default `3`).

- coef0:

  Constant term for the polynomial kernel (default `1`).

## Value

A function `K(xi, xj)` where `xi` and `xj` are numeric vectors of the
same length, returning a scalar kernel evaluation.

## Details

The three supported kernels are:

- **RBF:** `K(xi, xj) = exp(-‖xi - xj‖² / (2 * sigma²))`

- **Linear:** `K(xi, xj) = xi · xj`

- **Polynomial:** `K(xi, xj) = (xi · xj + coef0)^degree`

RBF and even-degree polynomial kernels satisfy Assumption 3 of the paper
(kernel symmetry), making them compatible with the symmetric models. The
linear kernel and odd-degree polynomial kernels do **not** satisfy
Assumption 3 and should not be used with
[`mape_sym_svr()`](https://pbenavidesh.github.io/psvr/reference/mape_sym_svr.md)
or
[`rmspe_sym_lssvr()`](https://pbenavidesh.github.io/psvr/reference/rmspe_sym_lssvr.md).

## Examples

``` r
K <- make_kernel("rbf", sigma = 0.5)
K(c(1, 2), c(3, 4))
#> [1] 1.125352e-07

K_lin <- make_kernel("linear")
K_lin(c(1, 0), c(0, 1))
#> [1] 0

K_poly <- make_kernel("polynomial", degree = 2, coef0 = 1)
K_poly(c(1, 2), c(3, 4))
#> [1] 144
```
