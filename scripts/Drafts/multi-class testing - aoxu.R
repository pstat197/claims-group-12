source('https://raw.githubusercontent.com/pstat197/pstat197a/main/materials/scripts/package-installs.R')

# packages
library(tidyverse)
library(tidymodels)
library(modelr)
library(Matrix)
library(sparsesvd)
library(glmnet)

# path to activity files on repo
url <- 'https://raw.githubusercontent.com/pstat197/pstat197a/main/materials/activities/data/'

# load a few functions for the activity
source(paste(url, 'projection-functions.R', sep = ''))

# read in data
claims <- paste(url, 'claims-multi-tfidf.csv', sep = '') %>%
  read_csv()

# preview
claims

# 1
# partition data
set.seed(102722)
partitions <- claims %>% initial_split(prop = 0.8)

# separate DTM from labels
test_dtm <- testing(partitions) %>%
  select(-.id, -bclass, -mclass)
test_labels <- testing(partitions) %>%
  select(.id, bclass, mclass)

# same, training set
train_dtm <- training(partitions) %>%
  select(-.id, -bclass, -mclass)
train_labels <- training(partitions) %>%
  select(.id, bclass, mclass)

# find projections based on training data
proj_out <- projection_fn(.dtm = train_dtm, .prop = 0.7)
train_dtm_projected <- proj_out$data

# how many components were used?
proj_out$n_pc

train <- train_labels %>%
  transmute(bclass = factor(bclass)) %>%
  bind_cols(train_dtm_projected)

fit <- glm(..., data = train, ...)

# store predictors and response as matrix and vector
x_train <- train %>% select(-bclass) %>% as.matrix()
y_train <- train_labels %>% pull(bclass)

# fit enet model
alpha_enet <- 0.3
fit_reg <- glmnet(x = x_train, 
                  y = y_train, 
                  family = 'binomial',
                  alpha = alpha_enet)

# choose a strength by cross-validation
set.seed(102722)
cvout <- cv.glmnet(x = x_train, 
                   y = y_train, 
                   family = 'binomial',
                   alpha = alpha_enet)

# store optimal strength
lambda_opt <- cvout$lambda.min

# view results
cvout
## Step 1: multinomial regression
#### get multiclass labels
y_train_multi <- train_labels %>% pull(mclass)

#### fit enet model
alpha_enet <- 0.2
fit_reg_multi <- glmnet(x = x_train, 
                        y = y_train_multi, 
                        family = 'multinomial',
                        alpha = alpha_enet)

#### choose a strength by cross-validation
set.seed(102722)
cvout_multi <- cv.glmnet(x = x_train, 
                         y = y_train_multi, 
                         family = 'multinomial',
                         alpha = alpha_enet)

#### view results
cvout

##Step 2: predictions

preds_multi <- predict(fit_reg_multi, 
                       s = cvout_multi$lambda.min, 
                       newx = x_test,
                       type = 'response')

as_tibble(preds_multi[, , 1]) 

pred_class <- as_tibble(preds_multi[, , 1]) %>% 
  mutate(row = row_number()) %>%
  pivot_longer(-row, 
               names_to = 'label',
               values_to = 'probability') %>%
  group_by(row) %>%
  slice_max(probability, n = 1) %>%
  pull(label)

pred_tbl <- table(pull(test_labels, mclass), pred_class)

pred_tbl