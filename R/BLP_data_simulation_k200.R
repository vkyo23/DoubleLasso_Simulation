# NOTE
## - Only first 4 variables of X are important for demand
## - Only first 2 variables of X are important for prices
## - Only first 4 variables of X are important for instrument

# 0. Setting -----
suppressMessages(library(dplyr))
suppressMessages(library(tidyr))
rm(list = ls())
Rcpp::sourceCpp('src/calc_share.cpp')
set.seed(1)
J <- 10
T <- 10
K <- 200
I <- 100
N <- J * T

market_id <- rep(1:T, each = J)

# 1. Parameters -----
## 1.1. Hyper parameter -----
Xi_sd <- 1/4

## 1.2. beta mean -----
beta_bar <- c(c(-2, 4, 4, 2, 2), rep(0, K - 4))

## 1.3. beta sd -----
sigma <- rep(0.5, K + 1)

## 1.4. alpha -----
alpha <- 5.0

## 1.5. Coefficient for prices -----
eta <- c(c(1, 1, 1), rep(0, K - 2))


## 1.6. Covariance matrix for X -----
cov <- matrix(NA, K, K)
for (i in 1:K) {
  for (j in 1:K) {
    cov[i, j] <- 0.5^(abs(i - j))
  }
}


# 2. Generate 100 datasets -----
for (s in 1:100) {
  
  # Covariates
  X <- mvtnorm::rmvnorm(N, mean = rep(0, K), sigma = cov)
  X <- cbind(1, X)
  colnames(X) <- paste0('X', 1:(K+1))
  
  # Unobserved market-product FE
  Xi <- Xi_sd * rnorm(N)
  
  # Price shifter
  C <- abs(rnorm(N, 1, 0.2))
  
  # Instrument
  Z <- runif(N)
  
  # Price
  P <- as.matrix(Xi + Z + X %*% eta + C)
  P <- ifelse(P < 0, abs(P), P)
  colnames(P) <- 'P'
  
  Z <- cbind(Z, C)
  colnames(Z) <- paste0('Z', 1:ncol(Z))
  
  # beta_i
  beta_i <- matrix(0, I, K + 1)
  for (k in 1:5) {
    beta_i[, k] <- rnorm(I, beta_bar[k], sigma[k])
  }
  
  S <- calc_share(P, X, alpha, beta_i, market_id)
  colnames(S) <- 'S'
  
  # Aggregate
  d <- expand_grid(j = 1:J, t = 1:T) %>% 
    arrange(t) %>% 
    bind_cols(
      S, P, X, Z
    ) %>% 
    mutate(I = I) %>% 
    mutate(simulation = s, .before = j)
  d <- d %>% 
    group_by(t) %>% 
    mutate(outside = 1 - sum(S)) %>% 
    ungroup() %>% 
    mutate(log_share = log(S / outside), .after = S)
  
  if (s == 1) {
    data <- d
  } else {
    data <- bind_rows(data, d)
  }
}

write.csv(data, 'data/simdata_K200.csv', row.names = FALSE)
