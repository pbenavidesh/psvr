# RBF sigma parameter for psvr models

A dials parameter for the RBF kernel bandwidth in psvr models. The
default range `[-3, 1]` on the log10 scale is a conservative fallback.
For best results, override the range using
[`sigma_heuristic()`](https://pbenavidesh.github.io/psvr/reference/sigma_heuristic.md)
computed on the preprocessed training data:

## Usage

``` r
rbf_sigma_psvr(range = c(-3, 1), trans = scales::log10_trans())
```

## Arguments

- range:

  Numeric vector of length 2 on the log10 scale. Default `c(-3, 1)`.

- trans:

  A `scales` transformation object. Default
  [`scales::log10_trans()`](https://scales.r-lib.org/reference/transform_log.html).

## Value

A `quant_param` dials object.

## Details

    train_baked <- rec |> prep() |> bake(new_data = train)
    sigma_med   <- sigma_heuristic(train_baked |> select(-outcome))
    rbf_sigma_custom <- rbf_sigma_psvr(
      range = c(log10(sigma_med / 10), log10(sigma_med * 10))
    )
    # Then inject via option_add():
    wf_set |> option_add(
      param_info = extract_parameter_set_dials(wf) |>
        update(rbf_sigma = rbf_sigma_custom),
      id = "your_workflow_id"
    )

## See also

[`sigma_heuristic()`](https://pbenavidesh.github.io/psvr/reference/sigma_heuristic.md)

## Examples

``` r
rbf_sigma_psvr()
#> RBF Sigma (psvr) (quantitative)
#> Transformer: log-10 [1e-100, Inf]
#> Range (transformed scale): [-3, 1]

# Override with data-driven range:
X <- matrix(rnorm(200), ncol = 4)
sigma_med <- sigma_heuristic(X)
rbf_sigma_psvr(range = c(log10(sigma_med / 10), log10(sigma_med * 10)))
#> RBF Sigma (psvr) (quantitative)
#> Transformer: log-10 [1e-100, Inf]
#> Range (transformed scale): [-0.599, 1.4]
```
