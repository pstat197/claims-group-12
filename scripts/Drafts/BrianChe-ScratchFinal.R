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

claims
word_tokens

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
#  Fit a logistic regression model with the PC’s obtained in the previous step as predictors and the class labels as response
train <- train_labels %>%
  transmute(bclass = factor(bclass)) %>%
  bind_cols(train_dtm_projected)

fit <- glm(formula=bclass~., data = train, family='binomial')

summary(fit)

# Compute and store predictions
probabilities <- fit %>% predict(train, type = "response")
predicted.classes <- ifelse(probabilities > 0.5, "pos", "neg")
mean(predicted.classes == train$bclass)

# Repeat 1-2 but using bigram tokenization in step 1

# Repeat step 3 using the bigram PCs, but add to your model the predictions obtained in step 4 as an offset

#################

