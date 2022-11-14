# Binary class prediction
# packages
library(tidyverse)
library(tidymodels)
library(modelr)
library(Matrix)
library(sparsesvd)
library(glmnet) 

# load functions
url <- 'https://raw.githubusercontent.com/pstat197/pstat197a/main/materials/activities/data/'
source(paste(url, 'projection-functions.R', sep = ''))

# load cleaned data
load('data/claims-clean-example.RData')

# partition data
set.seed(111322)
partitions <- claims_clean %>% initial_split(prop = 0.8)

# separate DTM from labels
test_dtm <- testing(partitions) %>%
  select(-.id, -bclass, -mclass)
test_labels <- testing(partitions) %>%
  select(.id, bclass)

# same, training set
train_dtm <- training(partitions) %>%
  select(-.id, -bclass, -mclass)
train_labels <- training(partitions) %>%
  select(.id, bclass)

# find projections based on training data
proj_out <- projection_fn(.dtm = train_dtm, .prop = 0.7)
train_dtm_projected <- proj_out$data

# how many components were used?
proj_out$n_pc