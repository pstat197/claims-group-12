
#  # library(openssl)
# 
# 
# py_config()
# 
# library(tensorflow)
# tf$constant('Hello world')
# 
# reticulate::conda_version()
# sessionInfo()
# 
# # conda_create("r-reticulate")
# # use_condaenv("r-reticulate")


# Task 2: Perform a secondary tokenization of the data to obtain bigrams. Fit a logistic principal component regression model to
# the word-tokenized data, and then input the predicted log-odds-ratios together with some number of principal components of the
# bigram-tokenized data to a second logistic regression model. Based on the results, does it seem like the bigrams capture additional 
# information about the claims status of a page?


source('https://raw.githubusercontent.com/pstat197/pstat197a/main/materials/scripts/package-installs.R')

# packages
library(tidyverse)
library(tidytext)
library(tokenizers)
library(textstem)
library(stopwords)
library(tidymodels)
library(dplyr)

library(keras)
library(tensorflow)
library(reticulate)

# data location
url <- 'https://raw.githubusercontent.com/pstat197/pstat197a/main/materials/labs/lab6-nn/data/claims-clean.csv'

# read in data
clean <- read_csv(url)
clean


# perform word tokenization
word_tokens <- clean$text_clean %>% tokenize_words

# perform tokenization of the data to obtain bigrams
bigram_tokens <- clean$text_clean %>% tokenize_ngrams(n = 2)

word_tokens
bigram_tokens

# obtaining an (n x p) document term matrix for the word tokens (predictors) 
clean$bclass


# obtain an (n x 1) vector of binary class labels (response).

#################
# partition data
set.seed(102722)
partitions <- clean %>% initial_split(prop = 0.8)

# separate DTM from labels
test_dtm <- testing(partitions) %>%
  select(-.id, -bclass)
test_labels <- testing(partitions) %>%
  select(.id, bclass)



# same, training set
train_dtm <- training(partitions) %>%
  select(-.id, -bclass)
train_labels <- training(partitions) %>%
  select(.id, bclass)

test_dtm
train_dtm

# find projections based on training data
proj_out <- projection_fn(.dtm = train_dtm, .prop = 0.7)
train_dtm_projected <- proj_out$data

train <- train_labels %>%
  transmute(bclass = factor(bclass)) %>%
  bind_cols(train_dtm_projected)

fit <- glm(bclass ~ ., data = train, 'binomial')


x_train <- train %>% select(-bclass) %>% as.matrix()
y_train <- train_labels %>% pull(bclass)

#################

# Fit a logistic principal component regression model to the word-tokenized data




# ========================================================================================== #
# data partitioning
# partition
set.seed(102722)
partitions <- clean %>%
  mutate(text_clean = str_trim(text_clean)) %>%
  filter(str_length(text_clean) > 5) %>%
  initial_split(prop = 0.8)

partitions

# preprocess the training partition into a TF-IDF document term matrix (DTM)
train_dtm <- training(partitions) %>%
  unnest_tokens(output = 'token', 
                input = text_clean) %>%
  group_by(.id, bclass) %>%
  count(token) %>%
  bind_tf_idf(term = token, 
              document = .id, 
              n = n) %>%
  pivot_wider(id_cols = c(.id, bclass), 
              names_from = token, 
              values_from = tf_idf,
              values_fill = 0) %>%
  ungroup()

train_dtm
train_dtm$bclass

table(train_dtm$bclass)

vec <- pull(train_dtm, bclass)
vec  

train_dtm$bclass

