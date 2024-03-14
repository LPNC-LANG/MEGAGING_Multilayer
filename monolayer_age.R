################################################################################
# Written by Clément Guichet, PhD Student
# LPNC - CNRS UMR 5105
# 2024

################################################################################
# MONOLAYER ----

rm(list=ls())

library(tidyverse)
library(data.table)
library(rio)
library(R.matlab)

data_monolayer <- readMat("H:/MEGAGING/output/eigenvector_centrality_monolayer.mat")
source("H:/MEGAGING/data/metadata.R")

analysis <- function(band) {
  frequency_band <<- case_when(band == 1 ~ "delta",
                              band == 2 ~ "theta",
                              band == 3 ~ "alpha",
                              band == 4 ~ "beta",
                              band == 5 ~ "gamma1",
                              band == 6 ~ "gamma2")
  print(frequency_band)
  data_EC <- data_monolayer$monolayer.EC[[band]] %>% as.data.frame()
  rownames(data_EC) <- subjects
  colnames(data_EC) <- regions
  
  # Elastic net ----
  # Logistic elastic net regression ----
  # https://bradleyboehmke.github.io/HOML/regularized-regression.html
  library(glmnet)
  library(caret)
  library(recipes)
  library(ROCR)
  library(vip)
  
  
  data_subjects_bu <- rio::import("H:/MEGAGING/data/list_subjects.xlsx") %>%
    merge(., data_EC %>% mutate(ID = rownames(.)), by = "ID") %>%
    mutate(gender = ifelse(gender == "M", -0.5, 0.5))
  
  data_full_tmp <- data_subjects_bu[, c(
    15:76 # Multilayer EC
  )] %>% scale() %>% as.data.frame() %>%
    cbind(
      age_group = data_subjects_bu$age_group,
      gender = data_subjects_bu$gender,
      TIV = as.numeric(scale(TIV)),
      MMSE = as.numeric(scale(data_subjects_bu$MMSE)),
      HAD_A = as.numeric(scale(data_subjects_bu$HAD_A)),
      HAD_D = as.numeric(scale(data_subjects_bu$HAD_D))
    ) %>%
    relocate(., age_group, .before = colnames(.)[1]) %>%
    relocate(., gender, .before = colnames(.)[2]) %>% 
    relocate(., TIV, .before = colnames(.)[3])
  
  # Regress out nuisance variables ----
  # Script converted from matlab script from Thomas Yeo's lab
  source("H:/MEGAGING/code/Multilayer_LP/CBIG_glm_regress_matrix.R")
  res <- CBIG_glm_regress_matrix(
    input_mtx = as.matrix(data_full_tmp[,4:65]),
    regressor = as.matrix(data_full_tmp[,c("TIV","gender","MMSE","HAD_A","HAD_D")]),
    polynomial_fit = 0)
  
  # This does the same thing
  # res_list = list()
  # for (feature in 4:65){
  #   mod <- lm(data_full_tmp[,feature] ~ 
  #               data_full_tmp$gender + 
  #               data_full_tmp$TIV + 
  #               data_full_tmp$MMSE + 
  #               HAD_A + HAD_D, 
  #             data = data_full_tmp)
  #   res_list[[feature]] <- mod$residuals
  # }
  # res_unlisted <- do.call(cbind, res_list) %>% as.data.frame()
  # colnames(res_unlisted) <- regions
  
  data_full_bu <- cbind(age_group = data_full_tmp$age_group, as.data.frame(res$resid_mtx))
  # Separate predictors and target variable
  x_bu <- model.matrix(age_group ~ ., data_full_bu)[, -1] # discard intercept
  y_bu <- data_full_bu$age_group %>%
    as.factor() %>%
    relevel(., ref = "Y")
  
  
  # Do 10 repeats of 5-fold stratified CV with a grid search across 50 values for λ and alpha
  nearZeroVar(data_full_bu, saveMetrics = TRUE)
  # Threshold-invariant approach ----
  # Set seed for reproducibility
  set.seed(123)
  # Train the model with caret::train
  cv_glmnet_bu <- caret::train(
    x = x_bu,
    y = y_bu,
    family = "binomial",
    method = "glmnet",
    trControl = trainControl(
      method = "repeatedcv",
      index = createFolds(factor(y_bu), 5, returnTrain = TRUE),
      number = 5,
      repeats = 10,
      search = "grid",
      summaryFunction = twoClassSummary, # Use AUC for evaluation
      classProbs = TRUE, # Enable class probabilities
      verboseIter = T
    ),
    metric = "ROC", # Use ROC as the primary metric
    tuneLength = 50
  )
  
  # Plot ROC curve
  library(pROC)
  cv_glmnet_prob_bu <- predict(cv_glmnet_bu, x_bu, "prob")$Y
  roc_curve <- roc(data_full_bu$age_group, cv_glmnet_prob_bu)
  auc_score <- auc(roc_curve)
  plot(roc_curve, col = "blue", lty = 1, main = "ROC Curve", xlab = "False Positive Rate", ylab = "True Positive Rate")
  
  # Print AUC
  cat("AUC:", auc_score, "\n")
  
  # Retrieve the best model's hyperparameters
  best_alpha <- cv_glmnet_bu$bestTune$alpha
  best_lambda <- cv_glmnet_bu$bestTune$lambda
  best_results <- cv_glmnet_bu$results %>%
    filter(alpha == best_alpha, lambda == best_lambda)
  
  # Print the best hyperparameters
  cat("Best Alpha:", best_alpha, "\n")
  cat("Best Lambda:", best_lambda, "\n")
  cat("Best Results:", "\n")
  print(best_results)
  
  # Fit the final model with the best hyperparameters
  elastic_bu <- glmnet(
    x_bu,
    y_bu,
    family = "binomial",
    alpha = best_alpha,
    lambda = best_lambda,
    standardize = FALSE  # the dataset was already preprocessed
  )
  
  # Visualize feature importance
  library(ggplot2)
  library(vip)
  vi <- vi(cv_glmnet_bu, method = "model")
  feature <- vi[1:5,]$Variable # Take the 5 most important variables
  
  vip_plot <- vip(cv_glmnet_bu, num_features = 5, geom = "col")
  vip_plot + theme_minimal() + theme(axis.text.x = element_text(angle = 45, hjust = 1)) + 
    labs(x = "Feature Importance", y = "Features") + ggtitle(frequency_band)
  
  # Print coefficients
  cat("Model Coefficients:\n")
  print(coef(elastic_bu)[feature,])
}

analysis(1) # delta
analysis(2) # theta
analysis(3) # alpha
analysis(4) # beta
analysis(5) # gamma1
analysis(6) # gamma2
