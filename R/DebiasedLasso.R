# 0. Setup -----
## Load packages
suppressMessages(library(tidyverse))
suppressMessages(library(glmnet))
suppressMessages(library(doParallel))
suppressMessages(library(doRNG))
suppressMessages(library(gt))
rm(list = ls())
set.seed(1)

## Clusters
cl <- makeCluster(8)
registerDoParallel(cl)
registerDoRNG(1)

## Load CV lasso function
cv_lasso <- function(formula, data) {
  formula <- as.formula(formula)
  md <- model.frame(formula, data)
  y <- md[, 1] %>% as.matrix()
  x <- md[, -1] %>% as.matrix()
  cv <- cv.glmnet(x = x, y = y, alpha = 1)
  out <- coef(glmnet(x = x, y = y, alpha = 1, lambda = cv$lambda.min)) %>% 
    as.matrix()
  nam <- rownames(out)
  out <- as.vector(out)
  names(out) <- nam
  return(out)
}

## DGP function
dgp <- function(N, K, tau, gamma, beta, Sigma) {
  
  ## Error term
  v <- rnorm(N)
  zeta <- rnorm(N)
  
  ## Controls 
  X <- mvtnorm::rmvnorm(N, mean = rep(0, K), sigma = Sigma)
  colnames(X) <- paste0('X_', 1:K)
  
  ## Treatment
  D <- X %*% gamma + v
  colnames(D) <- 'D'
  
  ## Outcome
  Y <- tau * D + X %*% beta + zeta
  colnames(Y) <- 'Y'
  
  ## Construct dataframe
  df <- bind_cols(Y, D, X)
  
  return(df)
}


# 1. Estimation K = 100 -----

## Sample size
N <- 100
K <- 100

## arameters
### True treatment effect
tau <- -3

### Coefficient for controls
gamma <- c(c(1, 1, 1, 0, 0), rep(0, K - 5))
beta <- c(c(4, 2, 2, 4, 2), rep(0, K - 5))

## VCOV of X
Sigma <- matrix(NA, K, K)
for (i in 1:K) {
  for (j in 1:K) {
    Sigma[i, j] <- 0.5^(abs(i - j))
  }
}

## Run
fit_k100 <- foreach(m = 1:100, .combine = rbind, .packages = c('tidyverse', 'glmnet')) %dopar% {
  
  ## Generating variables
  df <- dgp(N, K, tau, gamma, beta, Sigma)
  
  ## Post-Single Lasso estimator
  sel1 <- df %>% 
    select(Y, starts_with('X')) %>% 
    cv_lasso(Y ~ ., data = .)
  imp1 <- names(which(sel1 != 0))
  imp1 <- imp1[!str_detect(imp1, 'Intercept|D')]
  fit_ps <- df %>% 
    select(Y, D, all_of(imp1)) %>% 
    lm(Y ~ ., data = .)
  
  ## Debiased Lasso estimator
  sel1_deb <- df %>% 
    select(Y, starts_with('X')) %>% 
    cv_lasso(Y ~ ., data = .)
  imp1_deb <- names(which(sel1_deb != 0))
  sel2_deb <- df %>% 
    select(D, starts_with('X')) %>% 
    cv_lasso(D ~ ., data = .)
  imp2_deb <- names(which(sel2_deb != 0))
  imp_deb <- unique(c(imp1_deb, imp2_deb))
  imp_deb <- imp_deb[!str_detect(imp_deb, 'Intercept|D')]
  fit_deb <- df %>% 
    select(Y, D, all_of(imp_deb)) %>% 
    lm(Y ~ ., data = .)
  
  out <- tibble(`Post-Single` = coef(fit_ps)['D'],
                `Debiased` = coef(fit_deb)['D'])
  return(out)
}
fit_k100

# 2. Estimation K = 200 -----
## Sample size
N <- 100
K <- 200

## arameters
### True treatment effect
tau <- -3

### Coefficient for controls
gamma <- c(c(1, 1, 1, 0, 0), rep(0, K - 5))
beta <- c(c(4, 2, 2, 4, 2), rep(0, K - 5))

## VCOV of X
Sigma <- matrix(NA, K, K)
for (i in 1:K) {
  for (j in 1:K) {
    Sigma[i, j] <- 0.5^(abs(i - j))
  }
}

## Run
fit_k200 <- foreach(m = 1:100, .combine = rbind, .packages = c('tidyverse', 'glmnet')) %dopar% {
  
  ## Generating variables
  df <- dgp(N, K, tau, gamma, beta, Sigma)
  
  ## Post-Single Lasso estimator
  sel1 <- df %>% 
    select(Y, starts_with('X')) %>% 
    cv_lasso(Y ~ ., data = .)
  imp1 <- names(which(sel1 != 0))
  imp1 <- imp1[!str_detect(imp1, 'Intercept|D')]
  fit_ps <- df %>% 
    select(Y, D, all_of(imp1)) %>% 
    lm(Y ~ ., data = .)
  
  ## Debiased Lasso estimator
  sel1_deb <- df %>% 
    select(Y, starts_with('X')) %>% 
    cv_lasso(Y ~ ., data = .)
  imp1_deb <- names(which(sel1_deb != 0))
  sel2_deb <- df %>% 
    select(D, starts_with('X')) %>% 
    cv_lasso(D ~ ., data = .)
  imp2_deb <- names(which(sel2_deb != 0))
  imp_deb <- unique(c(imp1_deb, imp2_deb))
  imp_deb <- imp_deb[!str_detect(imp_deb, 'Intercept|D')]
  fit_deb <- df %>% 
    select(Y, D, all_of(imp_deb)) %>% 
    lm(Y ~ ., data = .)
  
  out <- tibble(`Post-Single` = coef(fit_ps)['D'],
                `Debiased` = coef(fit_deb)['D'])
  return(out)
}
fit_k200

# 3. Estimation true model -----
## Sample size
N <- 100
K <- 5

## arameters
### True treatment effect
tau <- -3

### Coefficient for controls
gamma <- c(1, 1, 1, 0, 0)
beta <- c(4, 2, 2, 4, 2)

## VCOV of X
Sigma <- matrix(NA, K, K)
for (i in 1:K) {
  for (j in 1:K) {
    Sigma[i, j] <- 0.5^(abs(i - j))
  }
}

## Run
fit_true <- foreach(m = 1:100, .combine = rbind, .packages = c('tidyverse', 'glmnet')) %dopar% {
  
  ## Generating variables
  df <- dgp(N, K, tau, gamma, beta, Sigma)
  
  f <- df %>% 
    lm(Y ~ ., data = .)
  
  out <- tibble(True = coef(f)['D'])
  return(out)
}
fit_true

## Stopping cluster
stopCluster(cl)

# 3. Visualization -----
d <- fit_k100 %>% 
  pivot_longer(cols = everything()) %>% 
  mutate(K = 'K = 100') %>% 
  bind_rows(
    fit_k200 %>% 
      pivot_longer(cols = everything()) %>% 
      mutate(K = 'K = 200')
  ) 
pdf('output/debiased_lasso.pdf', width = 10, height = 6)
d %>% 
  bind_rows(
    fit_true %>% 
      pivot_longer(cols = everything()) %>% 
      mutate(K = 'K = 200') %>% 
      bind_rows(
        fit_true %>% 
          pivot_longer(cols = everything()) %>% 
          mutate(K = 'K = 100')
      )
  ) %>% 
  ggplot(aes(x = value, fill = name)) +
  geom_density(alpha = .5) +
  facet_wrap(~ K) +
  theme_bw() +
  geom_vline(xintercept = tau, linetype = 'dashed') +
  xlab(expression(Estimated~treatment~effect~tau)) +
  ylab('Density') +
  theme(legend.position = 'top',
        legend.title = element_blank(),
        legend.text = element_text(size = 12),
        axis.text = element_text(size = 10),
        axis.title = element_text(size = 13),
        strip.text.x = element_text(size = 10))
dev.off()

tab <- d %>% 
  group_by(name, K) %>% 
  summarise(Mean = mean(value),
            Bias = mean(abs(value - tau)),
            RMSE = sqrt(mean((value - tau)^2)),
            .groups = 'drop') %>% 
  mutate(
    across(Mean:RMSE, round, 2)
  ) %>% 
  gt(
    rowname_col = 'name',
    groupname_col = 'K'
  ) 
gtsave(tab, 'output/DL_table.tex')
