from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, accuracy_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
from xgboost import XGBRegressor, XGBClassifier

# Load the datasets
X_train_path = "H:/Shared drives/AI Design Tool/00-PG_folder/03-Furniture AI Model/Data/preprocessed/X_train_full.csv"
y_train_path = "H:/Shared drives/AI Design Tool/00-PG_folder/03-Furniture AI Model/Data/preprocessed/y_train_full.csv"

X_train = pd.read_csv(X_train_path)
y_train = pd.read_csv(y_train_path)

# Display basic information about the datasets
# X_train_info = X_train.info()
# y_train_info = y_train.info()
#
# # Show first few rows of each dataset
# X_train_head = X_train.head()
# y_train_head = y_train.head()


# Select numerical features for scaling (excluding categorical 0/1 features)
num_features = ["Room_Length", "Room_Width", "Door_X_Position", "Door_Y_Position", "Door_Width"]
scaler = StandardScaler()

# Normalize numerical features
X_train_scaled = X_train.copy()
X_train_scaled[num_features] = scaler.fit_transform(X_train[num_features])

# Display processed dataset
X_train_scaled.head()

# Splitting data into train and test sets (80% train, 20% test)
X_train_split, X_test_split, y_train_split, y_test_split = train_test_split(
    X_train_scaled, y_train, test_size=0.2, random_state=42
)

# Define target variables for regression and classification
regression_targets = ["Toilet_X_Position", "Toilet_Y_Position", "Sink_X_Position", "Sink_Y_Position",
                      "Bathtub_X_Position", "Bathtub_Y_Position"]
classification_targets = ["Toilet_Rotation", "Sink_Rotation", "Bathtub_Rotation"]


# Train regression models for position prediction
regressors = {}
for target in regression_targets:
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_split, y_train_split[target])
    regressors[target] = model

# Apply rotation mapping before training
rotation_mapping = {0: 0, 90: 1, 180: 2, 270: 3}
for target in classification_targets:
    y_train_split[target] = y_train_split[target].map(rotation_mapping)
    y_test_split[target] = y_test_split[target].map(rotation_mapping)
# Train classification models for rotation prediction
classifiers = {}
for target in classification_targets:
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_split, y_train_split[target])  # Now y_train_split contains 0, 1, 2, 3
    classifiers[target] = model





# Evaluate models on the test set
regression_mae = {target: mean_absolute_error(y_test_split[target], regressors[target].predict(X_test_split))
                  for target in regression_targets}

classification_accuracy = {target: accuracy_score(y_test_split[target], classifiers[target].predict(X_test_split))
                           for target in classification_targets}
print(regression_mae)
print(classification_accuracy)

# Feature Engineering: Adding Derived Features

# Calculate available wall space for fixtures (Room perimeter minus door width)
X_train_scaled["Wall_Space"] = 2 * (X_train["Room_Length"] + X_train["Room_Width"]) - X_train["Door_Width"]

# Calculate distance from door to the nearest wall (minimum of X or Y position relative to room dimensions)
X_train_scaled["Dist_Door_Left"] = X_train["Door_X_Position"]
X_train_scaled["Dist_Door_Right"] = X_train["Room_Width"] - (X_train["Door_X_Position"] + X_train["Door_Width"])
X_train_scaled["Dist_Door_Top"] = X_train["Room_Length"] - X_train["Door_Y_Position"]
X_train_scaled["Dist_Door_Bottom"] = X_train["Door_Y_Position"]

# Normalize fixture dimensions (relative to room size)
X_train_scaled["Toilet_Size_Ratio"] = (19 * 28) / (X_train["Room_Length"] * X_train["Room_Width"])
X_train_scaled["Sink_Size_Ratio"] = (30 * 20) / (X_train["Room_Length"] * X_train["Room_Width"])
X_train_scaled["Bathtub_Size_Ratio"] = (30 * 60) / (X_train["Room_Length"] * X_train["Room_Width"])

# Hyperparameter tuning for XGBoost Regression
xgb_params = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2]
}

best_regressors = {}
for target in regression_targets:
    grid_search = GridSearchCV(XGBRegressor(random_state=42), xgb_params, cv=3, scoring='neg_mean_absolute_error', n_jobs=-1)
    grid_search.fit(X_train_split, y_train_split[target])
    best_regressors[target] = grid_search.best_estimator_
print("Best Regressor: \n")
print(grid_search.best_estimator_)
print("#"*100+"\n")
# Hyperparameter tuning for XGBoost Classification
best_classifiers = {}
for target in classification_targets:
    grid_search = GridSearchCV(XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss'),
                               xgb_params, cv=3, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train_split, y_train_split[target])
    best_classifiers[target] = grid_search.best_estimator_

print("Best Classifier: \n")
print(grid_search.best_estimator_)
print("#"*100+"\n")
# Evaluate tuned models on the test set
regression_mae = {target: mean_absolute_error(y_test_split[target], best_regressors[target].predict(X_test_split))
                  for target in regression_targets}

classification_accuracy = {target: accuracy_score(y_test_split[target], best_classifiers[target].predict(X_test_split))
                           for target in classification_targets}
inverse_rotation_mapping = {0: 0, 1: 90, 2: 180, 3: 270}
for target in classification_targets:
    y_train_split[target] = y_train_split[target].map(inverse_rotation_mapping)
    y_test_split[target] = y_test_split[target].map(inverse_rotation_mapping)

print(regression_mae)
print(classification_accuracy)
