# Install multiple packages with one line
install.packages(c("readr", "dplyr", "caret", "randomForest", "corrplot", "ggplot2", "pROC", "e1071", "caTools", "stats", "reshape2", "xgboost"))

# Load the packages with one line using lapply
lapply(c("readr", "dplyr", "caret", "randomForest", "corrplot", "ggplot2", "pROC", "e1071", "caTools", "stats", "reshape2", "xgboost"), library, character.only = TRUE)

# Loading necessary libraries
library(readr)
library(dplyr)
library(skimr)
library(caret)
library(randomForest)
library(corrplot)
library(ggplot2)
library(pROC)
library(e1071)
library(caTools)
library(stats)
library(reshape2)
library(xgboost)

# Use styler to make the code look nice
install.packages("styler")
library(styler)
style_file("./GroupProject.R")

# Define column names and load datasets
col_names <- c("image_name", paste0("V", 2:1921))
trainingData <- read_csv("./training.csv", col_names = col_names, show_col_types = FALSE)
testingData <- read_csv("./testing.csv", col_names = col_names, show_col_types = FALSE)

skimmed_training <- skim(trainingData)
skimmed_testing <- skim(testingData)

head(trainingData)
head(testingData)

print(skimmed_training) # Similar to describe and summary
print(skimmed_testing)

# Remove the first column as it's not meaningful for model training
# First column is the image name
trainingData <- trainingData[, -1]
testingData <- testingData[, -1]

# Label encoding for 'V2'
# Which is encoding the name of the plane to a number
trainingData$V2_label_encoded <- as.integer(factor(trainingData$V2))
testingData$V2_label_encoded <- as.integer(factor(testingData$V2))

# Scale the features, excluding first and last variable
# Which is both the target variable column
preProcValues <- preProcess(trainingData[, -c(1, length(trainingData))],
  method = c("center", "scale")
)
x_train <- predict(preProcValues, trainingData[, -c(1, length(trainingData))])
x_test <- predict(preProcValues, testingData[, -c(1, length(testingData))])

# Apply PCA to reduce dimensionality
pca <- prcomp(x_train, center = TRUE, scale. = TRUE)

# Function to plot the Scree Plot
# To visualize the percentage of variance explained by each principal component
# TODO: NEED TO DETERMINE ELBOW MANUALLY
# Elbow will be found in blue (individual) line
plotScree <- function(pca) {
  var_explained <- pca$sdev^2 / sum(pca$sdev^2) * 100
  cum_var_explained <- cumsum(var_explained)

  # Limit to the first 100 components
  var_explained <- var_explained[1:20]
  cum_var_explained <- cum_var_explained[1:20]

  plot(var_explained,
    type = "b", pch = 19, xlab = "Principal Component",
    ylab = "Percentage of Variance Explained", main = "Scree Plot",
    ylim = c(0, max(cum_var_explained)), col = "blue"
  )
  points(cum_var_explained, type = "b", pch = 18, col = "red")
  legend("topright", legend = c("Individual", "Cumulative"), col = c("blue", "red"), pch = c(19, 18))
}

# Convert labels for target variable
y_train_encoded <- as.numeric(trainingData$V2_label_encoded) - 1
y_test_encoded <- as.numeric(testingData$V2_label_encoded) - 1


# Execute Scree Plot function right after PCA
plotScree(pca)
---------------------------------------------------------------------------------
  ## Still working on XG Boost Ignore it temporarily##

  # TODO: DOUBLED CODE, EXISTS ON TOP ALREADY
  # Convert labels for XGBoost compatibility (Ensure labels start from 0)
  y_train_encoded <- as.numeric(as.factor(trainingData$V2)) - 1
y_test_encoded <- as.numeric(as.factor(testingData$V2)) - 1

# Assuming PCA has been applied to x_train and x_test
# TODO: Determine the number of components to keep from plotScree()
# Then change it to predict(pca, x_train)[, 1:n_components]
x_train_pca <- predict(pca, x_train)
x_test_pca <- predict(pca, x_test)

# Prepare the DMatrix objects
dtrain <- xgb.DMatrix(data = x_train_pca, label = y_train_encoded)
dtest <- xgb.DMatrix(data = x_test_pca, label = y_test_encoded)

params_grid <- expand.grid(
  nrounds = c(100, 300, 500),
  eta = c(0.01, 0.1, 0.3),
  max_depth = c(3, 6, 9),
  stringsAsFactors = FALSE
)

# Initialize an empty list to store the results
results_list <- list()

# # After looping, you can examine 'results_list' to find the best performing set of parameters
#
# # x_train_pca <- predict(pca, x_train)
# # x_test_pca <- predict(pca, x_test)
# #
# # # Convert labels for XGBoost compatibility (Ensure labels start from 0)
# # y_train_encoded <- as.numeric(trainingData$V2_label_encoded) - 1
# # y_test_encoded <- as.numeric(testingData$V2_label_encoded) - 1
# #
# # # Convert the data into the format required by XGBoost
# # dtrain <- xgb.DMatrix(data = x_train_pca, label = y_train_encoded)
# # dtest <- xgb.DMatrix(data = x_test_pca, label = y_test_encoded)
# #
# # # Define XGBoost training parameters
# # params <- list(
# #   objective = "multi:softmax",
# #   num_class = length(unique(y_train_encoded)),
# #   eval_metric = "mlogloss",
# #   max_depth = 6,
# #   eta = 0.3
# # )
# #
# # # Train the XGBoost model
# # nrounds <- 300  # Adjust based on your dataset and requirement
# # xgb_model <- xgb.train(params = params, data = dtrain, nrounds = nrounds)
# #
# # # Make predictions on the test set
# # predictions <- predict(xgb_model, dtest)
# #
# # # Evaluate the model performance
# # confMat <- confusionMatrix(table(factor(predictions), factor(y_test_encoded)))
# # print(confMat)
# #
# # # Optional: Plotting ROC Curve for binary classification or selected class in multi-class classification
# # # This requires transforming predictions into a binary format for the specific class and the pROC package
# # # For multi-class, consider a one-vs-rest approach or focus on specific metrics like accuracy from the confusion matrix
# #
# # # Plot the ROC curve for a specific class if binary or chosen class
# # # This is an illustrative example. Adapt it to your specific needs
# #
# #   # Assuming binary classification focusing on one specific class
# #   roc_result <- roc(response = as.factor(y_test_encoded == 0), predictor = as.numeric(predictions == 0))
# #   plot(roc_result, main = "ROC Curve")

# ----------------------------------------------------
# ------------ Pre-processing -----------------------

# Importing the data set
col_names <- c("image_name", paste0("V", 2:1921))
trainData <- read_csv("./training.csv", col_names = col_names, show_col_types = FALSE)
testData <- read_csv("./testing.csv", col_names = col_names, show_col_types = FALSE)

# Remove first column (image file name, not meaningful)
trainData <- trainData[, -1]
testData <- testData[, -1]

# Get the first column (image_name column)
first_column <- trainData[, 1]
first_column1 <- testData[, 1]

# Remove image_column from the front
trainData <- trainData[, -1]
testData <- testData[, -1]

# Add the first column to the end of the data frames
trainData <- cbind(trainData, first_column)
testData <- cbind(testData, first_column1)

# Feature scaling
trainData[-1921] <- scale(trainData[-1921])
testData[-1921] <- scale(testData[-1921])

# Check for count of missing values (Both had no missing values)
sum(is.na(trainData))
sum(is.na(testData))

# ---------- Dimensional Reduction ----------------------

# Scree Plot to determine the number of components to keep
pca <- prcomp(trainData[-1921], center = TRUE, scale. = TRUE)

# Calculate the variance explained by each principal component
pca_var <- pca$sdev^2 / sum(pca$sdev^2) * 100
pca_var <- pca_var[1:30]

# Plot the Scree Plot
plot(pca_var, type = "b", pch = 19, xlab = "Principal Component", ylab = "Percentage of Variance Explained", main = "Scree Plot")

# Applying PCA
# pcaComp can be changed according to elbow
# pcaComp is set to 200 because somehow it has the best accuracy for SVM

# Variable for PCA
pcaCompNum <- 200

pca <- preProcess(x = trainData[-1921], method = "pca", pcaComp = pcaCompNum)
trainData_pca <- predict(pca, newdata = trainData)
trainData_pca <- trainData_pca[, c(2:(pcaCompNum + 1), 1)]
testData_pca <- predict(pca, newdata = testData)
testData_pca <- testData_pca[, c(2:(pcaCompNum + 1), 1)]

# ------------------ Naive Bayes Model ---------------------------

# Train the Naive Bayes model
classifier_nb <- naiveBayes(V2 ~ ., data = trainData_pca)

# Make predictions on the test data
y_pred <- predict(classifier_nb, newdata = testData_pca[, -(pcaCompNum + 1)])

# Making the Confusion Matrix
cm_nb <- table(testData_pca[, (pcaCompNum + 1)], y_pred)

# Calculate accuracy from confusion matrix
accuracy_nb <- sum(diag(cm_nb)) / sum(cm_nb)
print(paste("Accuracy:", accuracy_nb))

# ---------------------- Decision Tree Classification Model ----------------
library(rpart)
classifier_dt <- rpart(
  formula = V2 ~ .,
  data = trainData_pca
)

# Predicting the Test set results
y_pred <- predict(classifier_dt, newdata = testData_pca[, -(pcaCompNum + 1)], type = "class")

# Making the Confusion Matrix
cm_dt <- table(testData_pca[, (pcaCompNum + 1)], y_pred)
print(cm_dt)

# Calculate accuracy from confusion matrix
accuracy_dt <- sum(diag(cm_dt)) / sum(cm_dt)
print(paste("Accuracy:", accuracy_dt))

# ------------------------- SVM Model ---------------------------

# Convert dependent variable to factor
trainData_pca$V2 <- as.factor(trainData_pca$V2)

# Train the SVM model
classifier_svm <- svm(V2 ~ ., data = trainData_pca, kernel = "radial", gamma = 0.01, cost = 10)

# Make predictions on the test data
y_pred_svm <- predict(classifier_svm, newdata = testData_pca[, -(pcaCompNum + 1)])

# Making the Confusion Matrix
cm_svm <- table(testData_pca[, (pcaCompNum + 1)], y_pred_svm)
print(cm_svm)

# Calculate accuracy from confusion matrix
accuracy_svm <- sum(diag(cm_svm)) / sum(cm_svm)
print(paste("Accuracy:", accuracy_svm))
