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


multilayer_EC_diff_bu <- multilayer_EC_gen_bu - multilayer_EC_con_bu
rownames(multilayer_EC_diff_bu) <- subjects
colnames(multilayer_EC_diff_bu) <- regions

# Logistic elastic net regression ----
# https://bradleyboehmke.github.io/HOML/regularized-regression.html
library(glmnet)
library(caret)
library(recipes)

data_subjects_bu <- rio::import("H:/MEGAGING/data/list_subjects.xlsx") %>%
  merge(., multilayer_EC_diff_bu %>% mutate(ID = rownames(.)), by = "ID") %>%
  mutate(gender = ifelse(gender == "M", -0.5, 0.5)) # Gender

data_full_bu <- data_subjects_bu[, c(
  5, # MMSE
  7:8, # HAD A + D
  15:76 # Multilayer EC
)] %>%
  scale() %>%
  as.data.frame() %>%
  cbind(
    age_group = data_subjects_bu$age_group,
    gender = data_subjects_bu$gender
  ) %>%
  relocate(., age_group, .before = colnames(.)[1]) %>%
  relocate(., gender, .before = colnames(.)[2])

x_bu <- model.matrix(age_group ~ ., data_full_bu)[, -1] # discard intercept
y_bu <- data_full_bu$age_group %>%
  as.factor() %>%
  relevel(., ref = "Y")

# Do 10 repeats of 5-fold stratified CV with a grid search across 50 values for λ and alpha
set.seed(2)
nearZeroVar(data_full_bu, saveMetrics = TRUE)
cv_glmnet_bu <- caret::train(
  x = x_bu,
  y = y_bu,
  family = "binomial",
  method = "glmnet",
  trControl = trainControl(
    method = "repeatedcv",
    index = createFolds(factor(y_bu), 5, returnTrain = T),
    number = 5,
    repeats = 10
  ),
  tuneLength = 50
)

cv_glmnet_bu$results %>%
  filter(alpha == cv_glmnet_bu$bestTune$alpha, lambda == cv_glmnet_bu$bestTune$lambda)

confusionMatrix(
  data = relevel(predict(cv_glmnet_bu, x_bu, "raw"), ref = "Y"),
  reference = relevel(y_bu, ref = "Y"),
  mode = "everything"
)

cv_glmnet_prob_bu <- predict(cv_glmnet_bu, x_bu, "prob")$Y
pacman::p_load(ROCR)
perf_bu <- prediction(cv_glmnet_prob_bu, y_bu) %>%
  performance(measure = "tpr", x.measure = "fpr")
# Plot ROC curve
plot(perf_bu, col = "blue", lty = 1)


# Retrieve the model coefficients
elastic_bu <- glmnet(x_bu, y_bu,
  family = "binomial",
  alpha = cv_glmnet_bu$bestTune$alpha,
  lambda = cv_glmnet_bu$bestTune$lambda,
  standardize = F # the dataset was already preprocessed
)

# predict(elastic_bu, x_bu, type = "class")
# predict(elastic_bu, x_bu, type = "response")
# predict(elastic_bu, x_bu, type = "link")
# Feature importance
vip::vip(cv_glmnet_bu, num_features = 10, geom = "col") + ggtitle("Feature Importance")
coef(elastic_bu)

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
