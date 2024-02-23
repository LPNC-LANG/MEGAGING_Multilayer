################################################################################
# Written by Clément Guichet, PhD Student
# LPNC - CNRS UMR 5105
# 2024

# rmarkdown::render("multilayer.R")
################################################################################

rm(list = ls())

library(tidyverse)
library(data.table)
library(rio)
library(R.matlab)
library(muxViz)
library(igraph)
library(ggraph)

# MULTILAYER CALCULATION ----

multilayer_EC_list_gen_bu <- list()
multilayer_EC_list_con_bu <- list()
for (subj in seq(22)) {
  # Load the supra-adjacency matrices
  supra_gen <- readMat("H:/MEGAGING/output/supra_generation.mat")$supra.generation[, , subj] 
  # Compute the binary multilayer EC
  multilayer_EC_list_gen_bu[[subj]] <- muxViz::GetMultiEigenvectorCentrality(binarizeMatrix(supra_gen), 6, 62) %>% t()

  supra_con <- readMat("H:/MEGAGING/output/supra_control.mat")$supra.control[, , subj]
  multilayer_EC_list_con_bu[[subj]] <- muxViz::GetMultiEigenvectorCentrality(binarizeMatrix(supra_con), 6, 62) %>% t()
}

multilayer_EC_gen_bu <- rbindlist(lapply(multilayer_EC_list_gen_bu, as.data.table)) %>% as.data.frame()
multilayer_EC_con_bu <- rbindlist(lapply(multilayer_EC_list_con_bu, as.data.table)) %>% as.data.frame()

# (Optional visualization)
# block_tensor <- muxViz::SupraAdjacencyToBlockTensor(supra_gen, Layers = 6, Nodes = 62)
# layers <- muxViz::SupraAdjacencyToNetworkList(supra_gen, 6, 62)
# node_tensor <- muxViz::SupraAdjacencyToNodesTensor(supra_gen, 6, 62)
#
# muxViz::plot_multiplex(layers, layout = "auto",
#                        layer.colors = c("red", "blue", "green", "black", "orange", "purple"))


regions <- c(
  "caudalanteriorcingulate L",
  "caudalanteriorcingulate R",
  "caudalmiddlefrontal L",
  "caudalmiddlefrontal R",
  "cuneus L",
  "cuneus R",
  "entorhinal L",
  "entorhinal R",
  "fusiform L",
  "fusiform R",
  "inferiorparietal L",
  "inferiorparietal R",
  "inferiortemporal L",
  "inferiortemporal R",
  "insula L",
  "insula R",
  "isthmuscingulate L",
  "isthmuscingulate R",
  "lateraloccipital L",
  "lateraloccipital R",
  "lateralorbitofrontal L",
  "lateralorbitofrontal R",
  "lingual L",
  "lingual R",
  "medialorbitofrontal L",
  "medialorbitofrontal R",
  "middletemporal L",
  "middletemporal R",
  "paracentral L",
  "paracentral R",
  "parahippocampal L",
  "parahippocampal R",
  "parsopercularis L",
  "parsopercularis R",
  "parsorbitalis L",
  "parsorbitalis R",
  "parstriangularis L",
  "parstriangularis R",
  "pericalcarine L",
  "pericalcarine R",
  "postcentral L",
  "postcentral R",
  "posteriorcingulate L",
  "posteriorcingulate R",
  "precentral L",
  "precentral R",
  "precuneus L",
  "precuneus R",
  "rostralanteriorcingulate L",
  "rostralanteriorcingulate R",
  "rostralmiddlefrontal L",
  "rostralmiddlefrontal R",
  "superiorfrontal L",
  "superiorfrontal R",
  "superiorparietal L",
  "superiorparietal R",
  "superiortemporal L",
  "superiortemporal R",
  "supramarginal L",
  "supramarginal R",
  "transversetemporal L",
  "transversetemporal R"
)

subjects <- c(
  "bm_014",
  "ca_001",
  "ca_019",
  "cc_007",
  "cm_013",
  "co_006",
  "dm_022",
  "ds_021",
  "el_018",
  "gb_020",
  "gh_017",
  "gp_011",
  "gv_005",
  "lf_012",
  "lr_008",
  "mp_004",
  "pe_009",
  "pl_016",
  "pr_015",
  "ra_003",
  "re_002",
  "sg_010"
)

TIV <- c(
  1530.42,
  1417.24,
  1440.27,
  1470.88,
  1353.45,
  1480.66,
  1409.18,
  1400.15,
  1361.40,
  1482.63,
  1702.60,
  1633.97,
  1337.13,
  1413.97,
  1582.01,
  1538.63,
  1589.39,
  1399.34,
  1491.61,
  1582.49,
  1302.83,
  1427.94
)


multilayer_EC_diff_bu <- multilayer_EC_gen_bu - multilayer_EC_con_bu
rownames(multilayer_EC_diff_bu) <- subjects
colnames(multilayer_EC_diff_bu) <- regions

# Logistic elastic net regression ----
# https://bradleyboehmke.github.io/HOML/regularized-regression.html
library(glmnet)
library(caret)
library(recipes)
library(ROCR)
library(vip)

data_subjects_bu <- rio::import("H:/MEGAGING/data/list_subjects.xlsx") %>%
  merge(., multilayer_EC_diff_bu %>% mutate(ID = rownames(.)), by = "ID") %>%
  mutate(gender = ifelse(gender == "M", -0.5, 0.5))

data_full_bu <- data_subjects_bu[, c(
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
  relocate(., TIV, .after = colnames(.)[3])

# Separate predictors and target variable
x_bu <- model.matrix(age_group ~ ., data_full_bu)[, -1] # discard intercept
y_bu <- data_full_bu$age_group %>%
  as.factor() %>%
  relevel(., ref = "Y")


# Do 10 repeats of 5-fold stratified CV with a grid search across 50 values for λ and alpha
set.seed(2)
# nearZeroVar(data_full_bu, saveMetrics = TRUE)

# Threshold-dependent ----
# cv_glmnet_bu <- caret::train(
#   x = x_bu,
#   y = y_bu,
#   family = "binomial",
#   method = "glmnet",
#   trControl = trainControl(
#     method = "repeatedcv",
#     index = createFolds(factor(y_bu), 5, returnTrain = T),
#     number = 5,
#     repeats = 10,
#     # The AUC is used to evaluate the classifier to avoid having to make decisions about the classification threshold.
#     summaryFunction = twoClassSummary, 
#     classProbs = T,
#   ),
#   metric = "Accuracy",
#   tuneLength = 50
# )

# Evaluate model performance
# confusion_matrix <- confusionMatrix(
#   data = relevel(predict(cv_glmnet_bu, x_bu, "raw"), ref = "Y"),
#   reference = relevel(y_bu, ref = "Y"),
#   mode = "everything"
# )

# Threshold-invariant approach ----
# Set seed for reproducibility
set.seed(2)
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
    summaryFunction = twoClassSummary, # Use AUC for evaluation
    classProbs = TRUE # Enable class probabilities
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
vip_plot <- vip(cv_glmnet_bu, num_features = 3, geom = "col", title = "Top 3 Feature Importance")
vip_plot + theme_minimal() + theme(axis.text.x = element_text(angle = 45, hjust = 1)) + 
  labs(x = "Feature Importance", y = "Features")

# Print coefficients
cat("Model Coefficients:\n")
print(coef(elastic_bu))

data_full_bu %>%
  group_by(age_group) %>%
  rstatix::get_summary_stats(c(`entorhinal L`, `middletemporal R`, `fusiform L`), type = "full")

# Visualization with ggseg ----
library(ggseg)
library(ggsegDKT)

atlas_dkt <- ggsegDKT::dkt %>% as.data.frame()

ggplot() +
  geom_brain(
    atlas = atlas_dkt %>%
      mutate(
        ACTIVATION =
          ifelse(hemi == "left" & region == "entorhinal", 1,
            ifelse(hemi == "left" & region == "fusiform", 1,
              ifelse(hemi == "right" & region == "middle temporal", 2,
                0
              )
            )
          )
      ),
    # ))),
    mapping = aes(fill = as.factor(ACTIVATION)),
    position = position_brain("horizontal"),
    size = 0.5,
    color = "black",
    show.legend = F
  ) +
  scale_fill_manual(values = c("lightgrey", "darkorange", "darkgreen")) +
  theme_void()
