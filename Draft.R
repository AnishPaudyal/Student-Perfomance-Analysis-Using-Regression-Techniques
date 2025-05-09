# Chapter X Code - Student Performance Analysis
# Predicting G3 (regression), G3_category (multiclass classification), and Pass (binary classification)

# Load required libraries
library(tidyverse)    # Data manipulation and visualization
library(caret)        # Model training and evaluation
library(glmnet)       # LASSO and Ridge regression
library(pls)          # PCR
library(e1071)        # SVM and Naive Bayes
library(MASS)         # LDA and QDA
library(rpart)        # Decision Trees
library(randomForest) # Random Forest
library(gbm)          # Gradient Boosting
library(class)        # KNN
library(pROC)         # ROC curves
library(boot)         # Cross-validation
library(BART)         # Bayesian Additive Regression Trees

# Set seed for reproducibility
set.seed(123)

# Data Loading and Preprocessing
data <- read.csv("student-por.csv")

# Convert categorical variables to factors
categorical_vars <- c("school", "sex", "address", "famsize", "Pstatus", "Mjob", 
                      "Fjob", "reason", "guardian", "schoolsup", "famsup", "paid", 
                      "activities", "nursery", "higher", "internet", "romantic")
data[categorical_vars] <- lapply(data[categorical_vars], as.factor)

# Create response variables
data$G3_category <- cut(data$G3, breaks = c(-1, 10, 15, 20), labels = c("Low", "Medium", "High"))
data$Pass <- factor(ifelse(data$G3 >= 10, "Pass", "Fail"))

# Create separate datasets
# Predictors (columns 1 to 32, excluding G3, G3_category, Pass)
predictors <- data[, 1:32]

# Regression dataset (predicting G3)
data_reg <- cbind(predictors, G3 = data$G3)

# Classification dataset (predicting G3_category)
data_class_cat <- cbind(predictors, G3_category = data$G3_category)

# Classification dataset (predicting Pass)
data_class_bin <- cbind(predictors, Pass = data$Pass)

# Train-test split
train <- sample(1:nrow(data), 0.8 * nrow(data))
train_reg <- data_reg[train, ]
test_reg <- data_reg[-train, ]
train_class_cat <- data_class_cat[train, ]
test_class_cat <- data_class_cat[-train, ]
train_class_bin <- data_class_bin[train, ]
test_class_bin <- data_class_bin[-train, ]

# Regression Models (Predicting G3)
# Multiple Linear Regression
fit_mlr <- lm(G3 ~ ., data = train_reg)
summary(fit_mlr)
pred_mlr <- predict(fit_mlr, test_reg)
mse_mlr <- mean((test_reg$G3 - pred_mlr)^2)
cat("MLR MSE:", mse_mlr, "\n")

# Assumptions
par(mfrow = c(2, 2))
plot(fit_mlr)
par(mfrow = c(1, 1))

# LASSO Regression
x_train <- model.matrix(G3 ~ ., train_reg)[, -1]
y_train <- train_reg$G3
fit_lasso <- cv.glmnet(x_train, y_train, alpha = 1)
bestlam_lasso <- fit_lasso$lambda.min
x_test <- model.matrix(G3 ~ ., test_reg)[, -1]
pred_lasso <- predict(fit_lasso, s = bestlam_lasso, newx = x_test)
mse_lasso <- mean((test_reg$G3 - pred_lasso)^2)
cat("LASSO MSE:", mse_lasso, "\n")
plot(fit_lasso)


# Ridge Regression
fit_ridge <- cv.glmnet(x_train, y_train, alpha = 0)
bestlam_ridge <- fit_ridge$lambda.min
pred_ridge <- predict(fit_ridge, s = bestlam_ridge, newx = x_test)
mse_ridge <- mean((test_reg$G3 - pred_ridge)^2)
cat("Ridge MSE:", mse_ridge, "\n")
plot(fit_ridge)

# Principal Component Regression (PCR)
fit_pcr <- pcr(G3 ~ ., data = train_reg, scale = TRUE, validation = "CV")
summary(fit_pcr)
validationplot(fit_pcr, val.type = "MSEP")
pred_pcr <- predict(fit_pcr, test_reg, ncomp = which.min(fit_pcr$validation$PRESS))
mse_pcr <- mean((test_reg$G3 - pred_pcr)^2)
cat("PCR MSE:", mse_pcr, "\n")

# Support Vector Regression (SVR)
fit_svr_linear <- svm(G3 ~ ., data = train_reg, kernel = "linear", cost = 1)
fit_svr_radial <- svm(G3 ~ ., data = train_reg, kernel = "radial", cost = 1, gamma = 0.1)
pred_svr_linear <- predict(fit_svr_linear, test_reg)
pred_svr_radial <- predict(fit_svr_radial, test_reg)
mse_svr_linear <- mean((test_reg$G3 - pred_svr_linear)^2)
mse_svr_radial <- mean((test_reg$G3 - pred_svr_radial)^2)
cat("SVR Linear MSE:", mse_svr_linear, "\nSVR Radial MSE:", mse_svr_radial, "\n")

# BART for Regression
fit_bart <- gbart(train_reg[, 1:32], train_reg$G3, x.test = test_reg[, 1:32], ntree = 100)
pred_bart <- fit_bart$yhat.test.mean
mse_bart <- mean((test_reg$G3 - pred_bart)^2)
cat("BART MSE:", mse_bart, "\n")

# Classification Models (Predicting Pass)
# Logistic Regression
fit_logistic <- glm(Pass ~ ., data = train_class_bin, family = binomial)
probs_logistic <- predict(fit_logistic, test_class_bin, type = "response")
pred_logistic <- ifelse(probs_logistic > 0.5, "Pass", "Fail")
table(pred_logistic, test_class_bin$Pass)
accuracy_logistic <- mean(pred_logistic == test_class_bin$Pass)
cat("Logistic Regression Accuracy:", accuracy_logistic, "\n")
plot.roc(test_class_bin$Pass, probs_logistic, main = "ROC Curve for Logistic Regression", col = "blue")

# Classification Models (Predicting G3_category)
# Linear Discriminant Analysis (LDA)
fit_lda <- lda(G3_category ~ ., data = train_class_cat)
pred_lda <- predict(fit_lda, test_class_cat)$class
table(pred_lda, test_class_cat$G3_category)
accuracy_lda <- mean(pred_lda == test_class_cat$G3_category)
cat("LDA Accuracy:", accuracy_lda, "\n")

# Quadratic Discriminant Analysis (QDA)
fit_qda <- qda(G3_category ~ ., data = train_class_cat)
pred_qda <- predict(fit_qda, test_class_cat)$class
table(pred_qda, test_class_cat$G3_category)
accuracy_qda <- mean(pred_qda == test_class_cat$G3_category)
cat("QDA Accuracy:", accuracy_qda, "\n")

# Naive Bayes
fit_nb <- naiveBayes(G3_category ~ ., data = train_class_cat)
pred_nb <- predict(fit_nb, test_class_cat)
table(pred_nb, test_class_cat$G3_category)
accuracy_nb <- mean(pred_nb == test_class_cat$G3_category)
cat("Naive Bayes Accuracy:", accuracy_nb, "\n")

# K-Nearest Neighbors (KNN)
train_x_knn <- model.matrix(~ . - 1, train_class_cat[, 1:32])
test_x_knn <- model.matrix(~ . - 1, test_class_cat[, 1:32])
fit_knn <- knn(train_x_knn, test_x_knn, train_class_cat$G3_category, k = 5)
table(fit_knn, test_class_cat$G3_category)
accuracy_knn <- mean(fit_knn == test_class_cat$G3_category)
cat("KNN Accuracy:", accuracy_knn, "\n")

# Decision Tree
fit_tree <- rpart(G3_category ~ ., data = train_class_cat, method = "class")
summary(fit_tree)
plot(fit_tree)
text(fit_tree, cex = 0.7)
pred_tree <- predict(fit_tree, test_class_cat, type = "class")
table(pred_tree, test_class_cat$G3_category)
accuracy_tree <- mean(pred_tree == test_class_cat$G3_category)
cat("Decision Tree Accuracy:", accuracy_tree, "\n")

# Random Forest
fit_rf <- randomForest(G3_category ~ ., data = train_class_cat, importance = TRUE)
pred_rf <- predict(fit_rf, test_class_cat)
table(pred_rf, test_class_cat$G3_category)
accuracy_rf <- mean(pred_rf == test_class_cat$G3_category)
cat("Random Forest Accuracy:", accuracy_rf, "\n")
varImpPlot(fit_rf)

# Gradient Boosting
fit_gbm <- gbm(as.numeric(G3_category) ~ ., data = train_class_cat, distribution = "multinomial", n.trees = 100)
pred_gbm <- predict(fit_gbm, test_class_cat, n.trees = 100, type = "response")
gbm_class <- apply(pred_gbm, 1, which.max)
gbm_class <- factor(gbm_class, levels = 1:3, labels = levels(data$G3_category))
table(gbm_class, test_class_cat$G3_category)
accuracy_gbm <- mean(gbm_class == test_class_cat$G3_category)
cat("Gradient Boosting Accuracy:", accuracy_gbm, "\n")

# Support Vector Machine (SVM)
fit_svm <- svm(G3_category ~ ., data = train_class_cat, kernel = "radial", cost = 1)
pred_svm <- predict(fit_svm, test_class_cat)
table(pred_svm, test_class_cat$G3_category)
accuracy_svm <- mean(pred_svm == test_class_cat$G3_category)
cat("SVM Radial Accuracy:", accuracy_svm, "\n")

# Model Validation with Cross-Validation
train_control <- trainControl(method = "cv", number = 10)
fit_cv_mlr <- cv.glm(data = train_reg, glmfit = glm(G3 ~ ., data = train_reg), K = 10)
mse_cv_mlr <- fit_cv_mlr$delta[1]
cat("MLR 10-fold CV MSE:", mse_cv_mlr, "\n")

# Feature Importance
cat("\nRandom Forest Variable Importance:\n")
print(importance(fit_rf))
cat("\nLASSO Coefficients:\n")
print(coef(fit_lasso, s = bestlam_lasso))