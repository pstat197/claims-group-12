# Primary task a) binary class prediction

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
  pull(bclass) %>%
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
model <- keras_model_sequential() %>%
  preprocess_layer() %>%
  layer_dropout(0.50) %>%
  layer_dense(units = 25) %>%
  layer_dropout(0.15) %>%
  layer_dense(1) %>%
  layer_activation(activation = 'sigmoid')

summary(model)

# configure for training
model %>% compile(
  loss = 'binary_crossentropy',
  optimizer = optimizer_sgd(),
  metrics = 'binary_accuracy'
)

# train
history <- model %>%
  fit(train_text, 
      train_labels,
      validation_split = 0.3,
      epochs = 5)

## CHECK TEST SET ACCURACY HERE
model$weights
evaluation <- evaluate(model, test_text, test_labels)

# model has accuracy of 0.81 which is higher than 
# the accuracy of 0.77 we got in task 1

saveRDS(history, file = "results/preds-group[N].RData")
saveRDS(evaluation, file = "results/preds-group[N].RData")
