# Set the working directory
# setwd("C:/Users/ACER/Documents/GitHub/INFO411-Airplane-Recognition/")

# Install multiple packages with one line
# install.packages(c("readr", "dplyr", "caret", "randomForest", "corrplot", "ggplot2", "pROC", "e1071", "caTools", "stats", "reshape2", "xgboost"))

# Load the packages with one line using lapply
lapply(c("readr", "dplyr", "caret", "randomForest", "corrplot", "ggplot2", "pROC", "e1071", "caTools", "stats", "reshape2", "xgboost", "rpart"), library, character.only = TRUE)

# Use styler to make the code look nice
install.packages("styler")
library(styler)
style_file("./GroupProjectV2.R")

# -------------- Importing the data set ----------------------------------------

# Define column names and load data sets
col_names <- c("image_name", paste0("V", 2:1921))

# Read the training data set
training_data <- read_csv("training.csv", col_names = col_names, show_col_types = FALSE)

# Read the testing data set
testing_data <- read_csv("testing.csv", col_names = col_names, show_col_types = FALSE)


# ---------------- Pre-processing the data set ---------------------------------
# # Print unique values of V2
unique_values <- unique(training_data$V2)
print(unique_values)

# Get summary statistics
summary(training_data$V2)

# Check for missing values in the entire training_data data set
missing_values <- colSums(is.na(training_data))

# # Print the count of missing values for each column
# print("Number of missing values in each column:")
# print(missing_values)

head(training_data)
tail(training_data)

# Check the number of records in the dataset
record_count <- nrow(training_data)
print(record_count)

target_distribution <- table(training_data$V2)
print(target_distribution)

# Calculate the total number of unique models in V2
total_models <- length(unique(training_data$V2))

# Print the total number of models
print(paste("Total number of models in V2:", total_models))

unique_categories <- unique(training_data$V2)

# Define a custom color palette with length matching the number of unique categories
color_palette <- rainbow(length(unique_categories))


# Create a strip plot of V2
ggplot(training_data, aes(x = V2, y = 0, color = V2)) +
  geom_jitter(width = 0.2, aes(fill = V2), alpha = 0.6) +
  scale_fill_manual(values = color_palette) +
  labs(x = "V2", y = NULL) +
  ggtitle("V2 Flight Model Names") +
  theme_minimal()


# Separate features from image and model for both training and testing data
training_features <- training_data[, -(1:2)]
testing_features <- testing_data[, -(1:2)]

# Feature Scaling
# Standardize the features for both training and testing data
scaled_training_features <- scale(training_features)
scaled_testing_features <- scale(testing_features)
y_train <- training_data$V2
y_test <- testing_data$V2

# ----------------- Dimensional Reduction --------------------------------------

# Fit PCA on the training data
pca_model <- prcomp(scaled_training_features)

# Print summary of the PCA model
summary(pca_model)

# Calculate the eigenvalues
eigenvalues <- pca_model$sdev^2

# Plot the scree plot
plot(1:100, eigenvalues[1:100],
  type = "b",
  xlab = "Principal Component", ylab = "Variance Explained",
  main = "Scree Plot"
)

# -------------- Determine the number of principle components base on ----------
# --------------------- the correlation values and scree plot ------------------

# Store principal components
principal_components <- pca_model$x

# Calculate correlation between principal components and original variables
correlation_with_original <- cor(scaled_training_features, principal_components)

# Find the column with the maximum absolute correlation for each principal component
max_correlation_indices <- apply(abs(correlation_with_original), 2, which.max)

# Get the variable names corresponding to the maximum correlation indices
variable_names <- colnames(scaled_training_features)
best_correlated_variables <- variable_names[max_correlation_indices]

# Display the results
result_df <- data.frame(
  Principal_Component = 1:ncol(principal_components),
  Best_Correlated_Variable = best_correlated_variables,
  Correlation_Value = sapply(1:ncol(principal_components), function(i) correlation_with_original[max_correlation_indices[i], i])
)

print(result_df)
# As you can see from the result_df above, the correlation values of principle components
# starts to decrease after like 17.

# Choose the number of principal components to retain
# We will choose the number of principal components to retain based on the scree plot and the correlation values
pcaCompNum <- 17

# Transform the training and testing data using the same PCA model
reduced_training_features <- data.frame(predict(pca_model, newdata = scaled_training_features))[, 1:pcaCompNum]
combined_train <- cbind(reduced_training_features, label = y_train)
reduced_training_data <- data.frame(combined_train)

reduced_testing_features <- data.frame(predict(pca_model, newdata = scaled_testing_features))[, 1:pcaCompNum]
combined_test <- cbind(reduced_testing_features, label = y_test)
reduced_testing_data <- data.frame(combined_test)

# Rename targets to Class (Optional, but to standardize between the new train and test set)
colnames(reduced_training_data)[ncol(reduced_training_data)] <- "Model"
colnames(reduced_testing_data)[ncol(reduced_testing_data)] <- "Model"

# -----------   Data visualization: correlation plot    ------------------------

corr_matrix <- cor(reduced_training_data[, 1:pcaCompNum])
corrplot(corr_matrix)

# -------------------- SVM Classification Model --------------------------------

library(e1071)

# Train the SVM model
svm_model <- svm(as.factor(reduced_training_data[, ncol(reduced_training_data)]) ~ .,
  data = reduced_training_data
)

# Make predictions on the testing data using the SVM model
svm_pred <- predict(svm_model, newdata = reduced_testing_data)

# Create a confusion matrix
conf_matrix <- confusionMatrix(svm_pred, as.factor(reduced_testing_data$Model))

# Extract accuracy from the confusion matrix
accuracy <- conf_matrix$overall["Accuracy"]
print(paste("Accuracy: ", round(accuracy, 4)))

# ------------------------ Random Forest Model ---------------------------------

# Train the Random Forest model
rf_model <- randomForest(as.factor(reduced_training_data[, ncol(reduced_training_data)]) ~ .,
  data = reduced_training_data
)

# Make predictions on the testing data using the random forest model
rf_pred <- predict(rf_model, newdata = reduced_testing_data)

# Confusion Matrix for Random Forest predictions for accuracy
conf_matrix_rf <- confusionMatrix(rf_pred, as.factor(reduced_testing_data$Model))
accuracy_rf <- conf_matrix_rf$overall["Accuracy"]
print(paste("Accuracy: ", round(accuracy_rf, 4)))

# ---------------------- Naive Bayes Model ------------------------------------

# Naive Bayes Fitting and Predicting
nb_model <- naiveBayes(as.factor(reduced_training_data[, ncol(reduced_training_data)]) ~ ., data = reduced_training_data)
nb_pred <- predict(nb_model, newdata = reduced_testing_data)

# Naive Bayes confusion matrix to get accuracy
conf_matrix_nb <- confusionMatrix(nb_pred, as.factor(reduced_testing_data$Model))
accuracy_nb <- conf_matrix_nb$overall["Accuracy"]
print(paste("Accuracy: ", round(accuracy_nb, 4)))

#  -------------- SVM model after Fine-tuning ----------------------------------

# Define a grid of cost and gamma values to try
cost_values <- c(0.1, 1, 10) # Example values, you can adjust these
gamma_values <- c(0.001, 0.01, 0.1) # Example values, you can adjust these

# Initialize variables to store the best parameters and corresponding accuracy
best_accuracy <- 0
best_cost <- 0
best_gamma <- 0

# Nested loops to iterate over the parameter grid
for (cost in cost_values) {
  for (gamma in gamma_values) {
    # Train the SVM model
    svm_model <- svm(as.factor(reduced_training_data[, ncol(reduced_training_data)]) ~ .,
      data = reduced_training_data,
      kernel = "linear",
      cost = cost,
      gamma = gamma
    )

    # Make predictions on the testing data using the SVM model
    svm_pred <- predict(svm_model, newdata = reduced_testing_data)

    # Create a confusion matrix
    conf_matrix <- confusionMatrix(svm_pred, as.factor(reduced_testing_data$Model))

    # Extract accuracy from the confusion matrix
    accuracy <- conf_matrix$overall["Accuracy"]

    # Check if current accuracy is better than the best accuracy so far
    if (accuracy > best_accuracy) {
      best_accuracy <- accuracy
      best_cost <- cost
      best_gamma <- gamma
    }
  }
}

# Print the best parameters and corresponding accuracy
cat("Best Parameters:\n")
cat("Cost:", best_cost, "\n")
cat("Gamma:", best_gamma, "\n")
cat("Best Accuracy:", best_accuracy, "\n")

# --------------- Random Forest Model After Fine-Tuning ------------------------

# Define a grid of hyperparameters to search
ntree_values <- c(100, 500, 1000) # Number of trees
mtry_values <- c(2, 4, 6) # Number of variables randomly sampled as candidates at each split

# Initialize variables to store the best parameters and corresponding accuracy
best_accuracy_rf <- 0
best_ntree <- 0
best_mtry <- 0

# Nested loops to iterate over the parameter grid
for (ntree in ntree_values) {
  for (mtry in mtry_values) {
    # Train the Random Forest model
    rf_model <- randomForest(as.factor(reduced_training_data[, ncol(reduced_training_data)]) ~ .,
      data = reduced_training_data,
      ntree = ntree,
      mtry = mtry
    )

    # Make predictions on the testing data using the Random Forest model
    rf_pred <- predict(rf_model, newdata = reduced_testing_data)

    # Create a confusion matrix
    conf_matrix_rf <- confusionMatrix(rf_pred, as.factor(reduced_testing_data$Model))

    # Extract accuracy from the confusion matrix
    accuracy_rf <- conf_matrix_rf$overall["Accuracy"]

    # Check if current accuracy is better than the best accuracy so far
    if (accuracy_rf > best_accuracy_rf) {
      best_accuracy_rf <- accuracy_rf
      best_ntree <- ntree
      best_mtry <- mtry
    }
 }
}

# Print the best parameters and corresponding accuracy
cat("Best Parameters for Random Forest:\n")
cat("Number of Trees:", best_ntree, "\n")
cat("mtry (Number of variables randomly sampled at each split):", best_mtry, "\n")
cat("Best Accuracy for Random Forest:", best_accuracy_rf, "\n")
#
#
# #-------------------------------------XG Boost --------------------------------------------------

# Exclude V2 (target variable) from the feature matrices

reduced_training_matrix <- as.matrix(training_data[, -c(1, 2)])
reduced_testing_matrix <- as.matrix(testing_data[, -c(1, 2)])
#
# # Check the class of the matrices
class(reduced_training_matrix)
class(reduced_testing_matrix)
#
# # Convert labels for XGBoost compatibility (Ensure labels start from 0)
y_train_encoded <- as.numeric(factor(training_data$V2)) - 1
y_test_encoded <- as.numeric(factor(testing_data$V2)) - 1
#
# # Check the class of the encoded labels
class(y_train_encoded)
class(y_test_encoded)
#
# # Convert the data into the format required by XGBoost
dtrain <- xgb.DMatrix(data = reduced_training_matrix, label = y_train_encoded)
dtest <- xgb.DMatrix(data = reduced_testing_matrix, label = y_test_encoded)

# # Define XGBoost training parameters
params <- list(
  objective = "multi:softmax",
  num_class = length(unique(y_train_encoded)),
  eval_metric = "mlogloss",
  max_depth = 6,
  eta = 0.3
)
#
# # Train the XGBoost model
nrounds <- 300 # Adjust based on your dataset and requirement
xgb_model <- xgb.train(params = params, data = dtrain, nrounds = nrounds)
#
# # Make predictions on the test set
predictions <- predict(xgb_model, dtest)
#
# # Evaluate the model performance
confMat <- confusionMatrix(table(factor(predictions), factor(y_test_encoded)))
# print(confMat)
#
# # Print accuracy
accuracy <- confMat$overall["Accuracy"]
print(paste("Accuracy:", round(accuracy, 4)))

# -----------------------------XG Boost Fine-Tuned------------------------------

# Define XGBoost training parameters with fine-tuned 
fine_tuned_params <- list(
  objective = "multi:softmax",
  num_class = length(unique(y_train_encoded)),
  eval_metric = "mlogloss",
  max_depth = 8,  
  eta = 0.2      
)

# Train the fine-tuned XGBoost model
fine_tuned_nrounds <- 600  
fine_tuned_xgb_model <- xgb.train(params = fine_tuned_params, data = dtrain, nrounds = fine_tuned_nrounds)

# Make predictions on the test set
fine_tuned_predictions <- predict(fine_tuned_xgb_model, dtest)

# Evaluate the model performance
fine_tuned_confMat <- confusionMatrix(table(factor(fine_tuned_predictions), factor(y_test_encoded)))

# Print accuracy
fine_tuned_accuracy <- fine_tuned_confMat$overall["Accuracy"]
print(paste("Fine-tuned Accuracy:", round(fine_tuned_accuracy, 4)))

