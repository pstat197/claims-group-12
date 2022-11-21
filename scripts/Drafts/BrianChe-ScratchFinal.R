# packages
library(tidyverse)
library(tidymodels)
library(modelr)
library(Matrix)
library(sparsesvd)
library(glmnet)
library(tidytext)
library(tokenizers)
library(textstem)
library(stopwords)
library(dplyr)

# path to activity files on repo
url <- 'https://raw.githubusercontent.com/pstat197/pstat197a/main/materials/activities/data/'

# load a few functions for the activity
source(paste(url, 'projection-functions.R', sep = ''))
source('scripts/preprocessing.R')

# read in data
claims <- paste(url, 'claims-multi-tfidf.csv', sep = '') %>%
  read_csv()

# preview
claims

# data location
url <- 'https://raw.githubusercontent.com/pstat197/pstat197a/main/materials/labs/lab6-nn/data/claims-clean.csv'

# read in data
clean <- read_csv(url)
clean

#### STEP 1
# perform word tokenization
word_tokens <- clean$text_clean %>% tokenize_words

# perform tokenization of the data to obtain bigrams
bigram_tokens <- clean$text_clean %>% tokenize_ngrams(n = 2)

# obtaining an (n x p) document term matrix for the word tokens (predictors) 

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

# (n x p) document term matrix for the word tokens (predictors) 
test_dtm
train_dtm

# (n x 1) vector of binary class labels (response).
test_labels %>%
  select(bclass)
train_labels %>%
  select(bclass)


#### STEP 2
# Project the DTM onto a number of principal components k of your choosing
# find projections based on training data
proj_out <- projection_fn(.dtm = train_dtm, .prop = 0.7)
train_dtm_projected <- proj_out$data

# how many components were used?
proj_out$n_pc

# (n x k) matrix of PC values.
train_dtm_projected

#### STEP 3
# Fit a logistic regression model with the PC’s obtained in the previous step as predictors and the class labels as response
# store predictors and response as matrix and vector
train <- train_labels %>%
  transmute(bclass = factor(bclass)) %>%
  bind_cols(train_dtm_projected)

# fit <- glm(..., data = train, ...)

x_train <- train %>% select(-bclass) %>% as.matrix()
y_train <- train_labels %>% pull(bclass)

# fit enet model
alpha_enet <- 0.3
fit_reg <- glmnet(x = x_train, 
                  y = y_train, 
                  family = 'binomial',
                  alpha = alpha_enet)

#### STEP 4
# Compute and store predictions
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

# project test data onto PCs
test_dtm_projected <- reproject_fn(.dtm = test_dtm, proj_out)

# coerce to matrix
x_test <- as.matrix(test_dtm_projected)

# compute predicted probabilities
preds <- predict(fit_reg, 
                 s = lambda_opt, 
                 newx = x_test,
                 type = 'link')


# store predictions in a data frame with true labels
pred_df <- test_labels %>%
  transmute(bclass = factor(bclass)) %>%
  bind_cols(pred = as.numeric(preds)) %>%
  mutate(bclass.pred = factor(pred > 0.5, 
                              labels = levels(bclass)))


# define classification metric panel 
panel <- metric_set(sensitivity, 
                    specificity, 
                    accuracy, 
                    roc_auc)

# compute test set accuracy
pred_df %>% panel(truth = bclass, 
                  estimate = bclass.pred, 
                  pred, 
                  event_level = 'second')

#### STEP 5
# Repeat 1-2 but using bigram tokenization in step 1

#  Adjust preprocessing for a claims_bigrams version we will use
claims_bigrams <- clean %>% 
  nlp_fn2()

# partition data for claims_bigrams
set.seed(102722)
partitions_bigrams  <- claims_bigrams %>% initial_split(prop = 0.8)

# separate DTM from labels
test_bigrams_dtm <- testing(partitions_bigrams) %>%
  select(-.id, -bclass)
test_bigrams_labels <- testing(partitions_bigrams) %>%
  select(.id, bclass)

test_bigrams_dtm

# same, training set
train_bigrams_dtm <- training(partitions_bigrams) %>%
  select(-.id, -bclass)
train_bigrams_labels <- training(partitions_bigrams) %>%
  select(.id, bclass)

# (n x p) document term matrix for the word tokens (predictors) 
test_bigrams_dtm
train_bigrams_dtm

# (n x 1) vector of binary class labels (response).
test_bigrams_labels %>%
  select(bclass)
train_bigrams_labels %>%
  select(bclass)

# Project the DTM onto a number of principal components k of your choosing
# find projections based on training data
proj_bigrams_out <- projection_fn(.dtm = train_bigrams_dtm, .prop = 0.7)

train_bigrams_dtm_projected <- proj_bigrams_out$data

# how many components were used?
proj_bigrams_out$n_pc

# (n x k) matrix of PC values.
train_bigrams_dtm_projected

#### STEP 6
# Repeat step 3 using the bigram PCs, but add to your model the predictions obtained in step 4 as an offset


# Fit a logistic regression model with the PC’s obtained in the previous step as predictors and the class labels as response
# store predictors and response as matrix and vector
train_bigrams  <- train_bigrams_labels %>%
  transmute(bclass = factor(bclass)) %>%
  bind_cols(train_bigrams_dtm_projected)


x_train_bigrams <- train_bigrams  %>% select(-bclass) %>% as.matrix()
y_train_bigrams <- train_bigrams_labels %>% pull(bclass)

# fit  model
fit_bigrams <- glmnet(y = y_train_bigrams,
                      x = x_train_bigrams,
                      offest = preds, 
                      family = 'binomial')

#### STEP 7
# Compare predictive accuracy between the model in step 3 and the model in step 6

# choose a strength by cross-validation

set.seed(102722)
cvout <- cv.glmnet(x = x_train_bigrams, 
                   y = y_train_bigrams, 
                   family = 'binomial',
                   alpha = alpha_enet)

# store optimal strength
lambda_opt <- cvout$lambda.min

# view results
cvout

# project test data onto PCs
test_bigrams_dtm_projected <- reproject_fn(.dtm = test_bigrams_dtm, proj_bigrams_out)

# coerce to matrix
x_test_bigrams <- as.matrix(test_bigrams_dtm_projected)

# compute predicted probabilities
preds_bigram <- predict(fit_bigrams, 
                 s = lambda_opt, 
                 newx = x_test_bigrams,
                 type = 'link')

# store predictions in a data frame with true labels
pred_df <- test_bigrams_labels %>%
  transmute(bclass = factor(bclass)) %>%
  bind_cols(pred = as.numeric(preds_bigram)) %>%
  mutate(bclass.pred = factor(pred > 0.5, 
                              labels = levels(bclass)))

# define classification metric panel 
panel <- metric_set(sensitivity, 
                    specificity, 
                    accuracy, 
                    roc_auc)

# compute test set accuracy
pred_df %>% panel(truth = bclass, 
                  estimate = bclass.pred, 
                  pred, 
                  event_level = 'second')



test_bigrams_dtm_projected


