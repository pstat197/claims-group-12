library(keras)
library(tensorflow)

library(openssl)

library(reticulate)
py_config()
tf$constant('Hello world')
conda_create("r-reticulate")
use_condaenv("r-reticulate")
