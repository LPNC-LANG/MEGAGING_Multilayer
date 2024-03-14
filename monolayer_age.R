################################################################################
# Written by Clément Guichet, PhD Student
# LPNC - CNRS UMR 5105
# 2024

################################################################################
# MONOLAYER ----

rm(list = ls())

library(tidyverse)
library(data.table)
library(rio)
library(R.matlab)

source("H:/MEGAGING/data/metadata.R")

analysis <- function(band) {
  frequency_band <<- case_when(
    band == 1 ~ "delta",
    band == 2 ~ "theta",
    band == 3 ~ "alpha",
    band == 4 ~ "beta",
    band == 5 ~ "gamma1",
    band == 6 ~ "gamma2"
  )
  print(frequency_band)
  
  
  # IMPORT MONOLAYER EC DATA
  data_monolayer_gen <- readMat("H:/MEGAGING/output/eigenvector_centrality_monolayer_generation.mat")
  data_EC_gen <- data_monolayer_gen$monolayer.EC.generation[[band]] %>% as.data.frame()
  rownames(data_EC_gen) <- subjects
  colnames(data_EC_gen) <- regions
  
  data_monolayer_con <- readMat("H:/MEGAGING/output/eigenvector_centrality_monolayer_control.mat")
  data_EC_con <- data_monolayer_con$monolayer.EC.control[[band]] %>% as.data.frame()
  rownames(data_EC_con) <- subjects
  colnames(data_EC_con) <- regions

  data_subjects_gen_bu <- rio::import("H:/MEGAGING/data/list_subjects.xlsx") %>%
    merge(., data_EC_gen %>% mutate(ID = rownames(.)), by = "ID") %>% 
    mutate(condition = rep("generation"))
  data_subjects_con_bu <- rio::import("H:/MEGAGING/data/list_subjects.xlsx") %>%
    merge(., data_EC_con %>% mutate(ID = rownames(.)), by = "ID") %>% 
    mutate(condition = rep("control"))
  
  # Logistic elastic net regression ----
  # https://bradleyboehmke.github.io/HOML/regularized-regression.html
  library(glmnet)
  library(caret)
  library(recipes)
  library(ROCR)
  library(vip)
  
  data_full_tmp <- data_subjects_gen_bu %>%
    mutate(gender = ifelse(gender == "M", -0.5, 0.5)) %>% 
    .[, c(
      15:76, # Multilayer EC
      4 # gender
    )] %>%
    cbind(
      age_group = data_subjects_gen_bu$age_group,
      TIV = as.numeric(scale(TIV)),
      MMSE = as.numeric(scale(data_subjects_gen_bu$MMSE)),
      HAD_A = as.numeric(scale(data_subjects_gen_bu$HAD_A)),
      HAD_D = as.numeric(scale(data_subjects_gen_bu$HAD_D))
    ) %>% 
    relocate(., age_group, .before = colnames(.)[1]) %>%
    relocate(., gender, .before = colnames(.)[2]) %>%
    relocate(., TIV, .before = colnames(.)[3]) %>% 
    relocate(., gender, .before = colnames(.)[4])
  
  # Regress out nuisance variables ----
  res_list = list()
  for (feature in 4:65){
    mod <- lm(data_full_tmp[,feature] ~
                data_subjects_con_bu[,11+feature] + # regress out the effect of control
                data_full_tmp$gender +
                data_full_tmp$TIV +
                data_full_tmp$MMSE +
                HAD_A + HAD_D,
              data = data_full_tmp)
    res_list[[feature]] <- mod$residuals
  }
  res_unlisted <- do.call(cbind, res_list) %>% scale() %>% as.data.frame()
  colnames(res_unlisted) <- regions
  data_full_bu <- cbind(age_group = data_full_tmp$age_group, res_unlisted)
  
  # Separate predictors and target variable
  x_bu <- model.matrix(age_group ~ ., data_full_bu)[, -1] # discard intercept
  y_bu <<- data_full_bu$age_group %>%
    as.factor() %>%
    relevel(., ref = "Y")
  
  
  # Do 10 repeats of 5-fold stratified CV with a grid search across 50 values for λ and alpha
  # nearZeroVar(data_full_bu, saveMetrics = TRUE)
  # Threshold-invariant approach ----
  # Set seed for reproducibility
  set.seed(1653)
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
  
  
  # Model Diagnostics ----
  probabilities <<- predict(cv_glmnet_bu, newx = x_bu, type = "raw")
  roc_curve <- roc(case_when(y_bu == "O" ~ 1, .default = 0), 
                   case_when(probabilities == "O" ~ 1, .default = 0))
  
  # Plot ROC curve
  plot(roc_curve, main = "ROC Curve", col = "blue")
  
  # Fit the final model with the best hyperparameters
  elastic_bu <- glmnet(
    x_bu,
    y_bu,
    family = "binomial",
    alpha = best_alpha,
    lambda = best_lambda,
    standardize = FALSE # the dataset was already preprocessed
  )
  
  # Visualize feature importance
  library(ggplot2)
  library(vip)
  vi <- vi(cv_glmnet_bu)
  feature <- vi[1:5, ]$Variable # Take the 5 most important variables
  print(vi)
  quantile(as.numeric(unlist(vi[,2])))
  
  # Print coefficients
  cat("Model Coefficients:\n")
  print(coef(elastic_bu)[feature, ])
  plot(varImp(cv_glmnet_bu), top = 5)
}

analysis(1) # delta

library(ggseg)
library(ggsegDKT)

atlas_dkt <- ggsegDKT::dkt %>% as.data.frame()

ggplot() +
  geom_brain(
    atlas = atlas_dkt %>%
      mutate(
        ACTIVATION =
          case_when(
            hemi == "left" & region == "entorhinal" ~ 1, # increase EC with age
            hemi == "right" & region == "superior parietal" ~ 1,
            
            hemi == "left" & region == "caudal anterior cingulate" ~ 2, # decrease EC with age
            .default = 0
          )
      ),
    mapping = aes(fill = as.factor(ACTIVATION)),
    position = position_brain("horizontal"),
    size = 0.5,
    color = "black",
    show.legend = F
  ) +
  scale_fill_manual(values = c("lightgrey", "red", "darkblue")) +
  theme_void()

analysis(2) # theta

ggplot() +
  geom_brain(
    atlas = atlas_dkt %>%
      mutate(
        ACTIVATION =
          case_when(
            hemi == "right" & region == "transverse temporal" ~ 2,
            .default = 0
          )
      ),
    mapping = aes(fill = as.factor(ACTIVATION)),
    position = position_brain("horizontal"),
    size = 0.5,
    color = "black",
    show.legend = F
  ) +
  scale_fill_manual(values = c("lightgrey", "darkblue")) +
  theme_void()

analysis(3) # alpha

ggplot() +
  geom_brain(
    atlas = atlas_dkt %>%
      mutate(
        ACTIVATION =
          case_when(
            hemi == "left" & region == "middle temporal" ~ 1,
            
            hemi == "right" & region == "insula" ~ 2,
            hemi == "right" & region == "precuneus" ~ 2,
            hemi == "right" & region == "superior frontal" ~ 2,
            .default = 0
          )
      ),
    mapping = aes(fill = as.factor(ACTIVATION)),
    position = position_brain("horizontal"),
    size = 0.5,
    color = "black",
    show.legend = F
  ) +
  scale_fill_manual(values = c("lightgrey", "red", "darkblue")) +
  theme_void()

analysis(4) # beta

ggplot() +
  geom_brain(
    atlas = atlas_dkt %>%
      mutate(
        ACTIVATION =
          case_when(
            hemi == "right" & region == "fusiform" ~ 1, 
            
            hemi == "right" & region == "rostral middle frontal" ~ 2,
            hemi == "left" & region == "pars opercularis" ~ 2,
            hemi == "left" & region == "medial orbitofrontal" ~ 2,
            hemi == "right" & region == "superior frontal" ~ 2,
            .default = 0
          )
      ),
    mapping = aes(fill = as.factor(ACTIVATION)),
    position = position_brain("horizontal"),
    size = 0.5,
    color = "black",
    show.legend = F
  ) +
  scale_fill_manual(values = c("lightgrey", "red", "darkblue")) +
  theme_void()

analysis(5) # gamma1
analysis(6) # gamma2

ggplot() +
  geom_brain(
    atlas = atlas_dkt %>%
      mutate(
        ACTIVATION =
          case_when(
            hemi == "left" & region == "lateral orbitofrontal" ~ 2,
            hemi == "right" & region == "inferior temporal" ~ 2,
            hemi == "right" & region == "precuneus" ~ 2,
            hemi == "left" & region == "inferior temporal" ~ 2,
            hemi == "left" & region == "precentral" ~ 2,
            .default = 0
          )
      ),
    mapping = aes(fill = as.factor(ACTIVATION)),
    position = position_brain("horizontal"),
    size = 0.5,
    color = "black",
    show.legend = F
  ) +
  scale_fill_manual(values = c("lightgrey", "darkblue")) +
  theme_void()
