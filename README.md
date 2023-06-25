# Double-Lasso Simulation

In this repository, I provide some simulation codes for Double-Lasso models. The simulations include simple Double Selection Lasso (Belloni, Chernozhukov & Hansen 2014), and simpler version of BLP-2LASSO (Gillen, *et al*. 2019). The former simulation is done by only `R`, and the latter (BLP) is done by `C++`, `R` and `Python` (for MPEC with `Pyomo`). 

## Description

- Python: Directory of `Python` codes
- R: Directory of `R` codes
- data: Directory of simulation data
- output: Directory of simulation results
- src: Directory of `C++` codes
- `run.sh`: Code for implementing all simulation practices

## Implementation

In terminal, please run the following code:

```{sh}
sh run.sh
```

## Refrences

1. Belloni, A., Chernozhukov, V., & Hansen, C. (2014). "Inference on treatment effects after selection among high-dimensional controls". *The Review of Economic Studies*, 81(2), 608-650.

2. Gillen, B. J., Montero, S., Moon, H. R., & Shum, M. (2019). "BLP-2LASSO for aggregate discrete choice models with rich covariates". *The Econometrics Journal*, 22(3), 262-281.
