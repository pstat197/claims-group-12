library(keras)
library(tensorflow)
library(reticulate)
 # library(openssl)


py_config()

library(tensorflow)
tf$constant('Hello world')

reticulate::conda_version()
sessionInfo()

# conda_create("r-reticulate")
# use_condaenv("r-reticulate")
