
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

# packages
library(tidyverse)
library(tidytext)
library(tokenizers)
library(textstem)
library(stopwords)
library(tidymodels)

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
  