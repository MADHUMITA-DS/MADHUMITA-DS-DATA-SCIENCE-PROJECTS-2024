library(dplyr)
library(caret)
library(rpart)
library(e1071)

# Load data
vehicle_data <- read.csv("your path/vehicle path (Day).csv")
geo_data <- read.csv("your path/geo1.csv")

# Preprocessing - Handle missing values
vehicle_data <- na.omit(vehicle_data)

# Convert target variable to factor
vehicle_data$vehicle_type <- as.factor(vehicle_data$vehicle_type)

# Separate features and target
X <- vehicle_data %>% select(-vehicle_type)
y <- vehicle_data$vehicle_type

# One-hot encode categorical features, if any
X <- dummyVars(" ~ .", data = X) %>% predict(newdata = X) %>% as.data.frame()

# Train-test split
set.seed(42)
trainIndex <- createDataPartition(y, p = 0.8, list = FALSE)
X_train <- X[trainIndex, ]
X_test <- X[-trainIndex, ]
y_train <- y[trainIndex]
y_test <- y[-trainIndex]

# Initialize and train the Decision Tree model
dt_model <- rpart(y_train ~ ., data = data.frame(X_train, y_train), method = "class")

# Make predictions and evaluate
y_pred <- predict(dt_model, X_test, type = "class")
cat("Decision Tree Accuracy:", mean(y_pred == y_test), "\n")
cat("Decision Tree Classification Report:\n")
print(confusionMatrix(y_pred, y_test))
