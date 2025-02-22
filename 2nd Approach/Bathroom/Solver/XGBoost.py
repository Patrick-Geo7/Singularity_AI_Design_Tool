from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor, XGBClassifier

# Define target variables
regression_targets = ["Toilet_X_Position", "Toilet_Y_Position", "Sink_X_Position", "Sink_Y_Position",
                      "Bathtub_X_Position", "Bathtub_Y_Position"]
classification_targets = ["Toilet_Rotation", "Sink_Rotation", "Bathtub_Rotation"]

# Split the dataset into training and testing sets (80% train, 20% test)
X_train_split, X_test_split, y_train_split, y_test_split = train_test_split(
    X_train_scaled, y_train, test_size=0.2, random_state=42
)

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

# Hyperparameter tuning for XGBoost Classification
best_classifiers = {}
for target in classification_targets:
    grid_search = GridSearchCV(XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss'),
                               xgb_params, cv=3, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train_split, y_train_split[target])
    best_classifiers[target] = grid_search.best_estimator_

# Evaluate tuned models on the test set
regression_mae = {target: mean_absolute_error(y_test_split[target], best_regressors[target].predict(X_test_split))
                  for target in regression_targets}

classification_accuracy = {target: accuracy_score(y_test_split[target], best_classifiers[target].predict(X_test_split))
                           for target in classification_targets}

regression_mae, classification_accuracy
