# Primary task b) multi-class prediction

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

## PREPROCESSING
#################
source('scripts/preprocessing.R')

# load raw data
load('data/claims-raw.RData')

# preprocess (will take a minute or two)
claims_clean <- claims_raw %>%
  parse_data()

# export
save(claims_clean, file = 'data/claims-clean-example.RData')

## MODEL TRAINING (NN)
######################
require(tidyverse)
require(tidymodels)
require(keras)
require(tensorflow)

# load cleaned data
load('data/claims-clean-example.RData')

# partition
set.seed(111422)
partitions <- claims_clean %>%
  initial_split(prop = 0.7)

train_text <- training(partitions) %>%
  pull(text_clean)
train_labels <- training(partitions) %>%
  pull(mclass) %>%
  as.numeric() - 1

test_text <- testing(partitions) %>%
  pull(text_clean)
test_labels <- testing(partitions) %>%
  pull(bclass) %>%
  as.numeric() - 1

# create a preprocessing layer
preprocess_layer <- layer_text_vectorization(
  standardize = NULL,
  split = 'whitespace',
  ngrams = 2,
  max_tokens = NULL,
  output_mode = 'tf_idf'
)

preprocess_layer %>% adapt(train_text)


# changed layer_dropout from 0.25 to 0.75
# define NN architecture
model2 <- keras_model_sequential() %>%
  preprocess_layer() %>%
  layer_dropout(0.50) %>%
  layer_dense(units = 25) %>%
  layer_dropout(0.15) %>%
  layer_dense(1) %>%
  layer_activation(activation = 'sigmoid')

summary(model2)

# configure for training
model2 %>% compile(
  loss = 'binary_crossentropy',
  optimizer = optimizer_sgd(),
  metrics = 'binary_accuracy'
)

# train
history2 <- model2 %>%
  fit(train_text, 
      train_labels,
      validation_split = 0.3,
      epochs = 5)

## CHECK TEST SET ACCURACY HERE
model2$weights
evaluation2 <- evaluate(model2, test_text, test_labels)

# model has accuracy of 0.5981 which is smaller than 
# the accuracy of 0.77 we got in task 1

# Save an object to a file
saveRDS(history2, file = "results/preds-group[N].RData")
saveRDS(evaluation2, file = "results/preds-group[N].RData")

