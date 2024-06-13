#%%
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Load data
file_path = 'HR_data.csv'
data = pd.read_csv(file_path)

# Data overview
print(data.head())

# Features and target
X = data[['HR_Mean', 'HR_Median', 'HR_std', 'HR_Min', 'HR_Max', 'HR_AUC']]
y = data['Frustrated']
groups = data['Individual']

# Define the outer cross-validation method
outer_cv = GroupKFold(n_splits=5)

# Initialize lists to store results
baseline_accuracies = []
rf_accuracies = []
lr_accuracies = []

# Random Forest hyperparameters
rf_params = {
    'classifier__n_estimators': [50, 100, 200],
    'classifier__max_depth': [None, 10, 20, 30],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 4],
    'classifier__bootstrap': [True, False]
}

# Logistic Regression hyperparameters
lr_params = {
    'classifier__C': [0.01, 0.1, 1, 10, 100],
    'classifier__solver': ['newton-cg', 'lbfgs']
}

# Perform outer cross-validation
for train_idx, test_idx in outer_cv.split(X, y, groups):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    group_train, group_test = groups.iloc[train_idx], groups.iloc[test_idx]

    # Baseline model
    baseline = DummyClassifier(strategy='most_frequent')
    baseline.fit(X_train, y_train)
    baseline_pred = baseline.predict(X_test)
    baseline_accuracy = accuracy_score(y_test, baseline_pred)
    baseline_accuracies.append(baseline_accuracy)

    # Random Forest model
    rf_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(random_state=42))
    ])
    
    inner_cv = GroupKFold(n_splits=3)
    rf_grid_search = GridSearchCV(estimator=rf_pipeline, param_grid=rf_params, cv=inner_cv, n_jobs=-1, verbose=0)
    rf_grid_search.fit(X_train, y_train, groups=group_train)

    rf_best_model = rf_grid_search.best_estimator_
    rf_best_pred = rf_best_model.predict(X_test)
    rf_best_accuracy = accuracy_score(y_test, rf_best_pred)
    rf_accuracies.append(rf_best_accuracy)

    print(f'\nRandom Forest Fold Accuracy: {rf_best_accuracy}')
    print(f'\nRandom Forest Best Parameters: {rf_grid_search.best_params_}')
    print("\n", classification_report(y_test, rf_best_pred, zero_division=0))

    # Logistic Regression model
    lr_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(multi_class='multinomial', random_state=42, max_iter=2000))
    ])
    
    lr_grid_search = GridSearchCV(estimator=lr_pipeline, param_grid=lr_params, cv=inner_cv, n_jobs=-1, verbose=0)
    lr_grid_search.fit(X_train, y_train, groups=group_train)

    lr_best_model = lr_grid_search.best_estimator_
    lr_best_pred = lr_best_model.predict(X_test)
    lr_best_accuracy = accuracy_score(y_test, lr_best_pred)
    lr_accuracies.append(lr_best_accuracy)

    print(f'\nLogistic Regression Fold Accuracy: {lr_best_accuracy}')
    print(f'\nLogistic Regression Best Parameters: {lr_grid_search.best_params_}')
    print(classification_report(y_test, lr_best_pred, zero_division=0))

# Compare models
print(f'\nBaseline Mean Accuracy: {np.mean(baseline_accuracies)}')
print(f'\nRandom Forest Mean Accuracy: {np.mean(rf_accuracies)}')
print(f'\nLogistic Regression Mean Accuracy: {np.mean(lr_accuracies)}')
# %%
