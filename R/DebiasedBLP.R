suppressMessages(library(tidyverse))
suppressMessages(library(gt))
rm(list = ls())

# 1. Load data -----
## 1.1. K = 100 -----
### 1.1.1. Post-Single Lasso-BLP ------ 
ps100 <- read.table('output/result_single_K100.csv', sep = ',')
alpha_ps100 <- ps100[2, ] %>% 
  as.numeric()

### 1.1.2. Debiased Lasso-BLP -----
deb100 <- read.table('output/result_debiased_K100.csv', sep = ',')
alpha_deb100 <- deb100[2, ] %>% 
  as.numeric()

## 1.2. K = 200 -----
### 1.2.1. Post-Single Lasso-BLP ------ 
ps200 <- read.table('output/result_single_K200.csv', sep = ',')
alpha_ps200 <- ps200[2, ] %>% 
  as.numeric()

### 1.2.2. Debiased Lasso-BLP -----
deb200 <- read.table('output/result_debiased_K200.csv', sep = ',')
alpha_deb200 <- deb200[2, ] %>% 
  as.numeric()

## 1.3. True model -----
true <- read.table('output/result_true.csv', sep =',')
alpha_true <- true[2, ] %>% 
  as.numeric()

# 2. Visualization -----
d <- tibble(`Single` = alpha_ps100, Double = alpha_deb100) %>% 
  mutate(id = row_number()) %>% 
  pivot_longer(-id) %>% 
  mutate(K = 'K = 100', .before = name) %>% 
  bind_rows(
   tibble(`Single` = alpha_ps200, Double = alpha_deb200) %>%
     mutate(id = row_number()) %>%
     pivot_longer(-id) %>%
     mutate(K = 'K = 200', .before = name)
  )
pdf('output/BLP_Lasso.pdf', width = 10, height = 6)
d %>% 
  ggplot(aes(x = value, fill = name)) +
  geom_density(alpha = .5) +
  facet_wrap(~ K) +
  geom_vline(xintercept = 5, linetype = 'dashed') +
  theme_bw() +
  xlab(expression(Estimated~elasticity~alpha)) +
  ylab('Density') +
  scale_x_continuous(breaks = seq(-30, 30, 5)) +
  theme(legend.position = 'top',
        legend.title = element_blank(),
        legend.text = element_text(size = 12),
        axis.text = element_text(size = 10),
        axis.title = element_text(size = 13),
        strip.text.x = element_text(size = 10))
dev.off()

## Comparing true model to Double one
pdf('output/BLP_Lasso_deb_true.pdf', width = 10, height = 6)
d %>%  
  filter(name == 'Double') %>% 
  bind_rows(
  tibble(True = alpha_true) %>% 
    mutate(id = row_number()) %>% 
    pivot_longer(-id) %>% 
    mutate(K = 'K = 100', .before = name) %>% 
    bind_rows(
      tibble(True = alpha_true) %>% 
        mutate(id = row_number()) %>% 
        pivot_longer(-id) %>% 
        mutate(K = 'K = 200', .before = name)
    )
) %>% 
  ggplot(aes(x = value, fill = name)) +
  geom_density(alpha = .5) +
  facet_wrap(~ K) +
  geom_vline(xintercept = 5, linetype = 'dashed') +
  theme_bw() +
  xlab(expression(Estimated~elasticity~alpha)) +
  ylab('Density') +
  scale_x_continuous(breaks = seq(-30, 30, 1.25)) +
  theme(legend.position = 'top',
        legend.title = element_blank(),
        legend.text = element_text(size = 12),
        axis.text = element_text(size = 10),
        axis.title = element_text(size = 13),
        strip.text.x = element_text(size = 10))
dev.off()

# 3. Descriptive statistics -----
tab <- d %>% 
  group_by(K, name) %>% 
  summarise(Mean = mean(value),
            Bias = mean(abs(value - 5)),
            RMSE = sqrt(mean((value - 5)^2)),
            .groups = 'drop')  %>% 
  mutate(
    across(Mean:RMSE, round, 2)
  ) %>% 
  gt(
    rowname_col = 'name',
    groupname_col = 'K'
  ) 
gtsave(tab, 'output/BLP_table.tex')
