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