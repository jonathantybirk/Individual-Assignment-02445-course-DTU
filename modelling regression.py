#%%
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import sem, t, ttest_rel
import csv

# Load data
file_path = 'HR_data.csv'
data = pd.read_csv(file_path)

# Features and target
features = ['HR_Mean', 'HR_Median', 'HR_std', 'HR_Min', 'HR_Max', 'HR_AUC']
X = data[features]
y = data['Frustrated']
groups = data['Individual']

# Preprocessing
numeric_features = ['HR_Mean', 'HR_Median', 'HR_std', 'HR_Min', 'HR_Max', 'HR_AUC']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
    ])

# Initialize lists to store results and parameters
results = {'baseline': [], 'ridge': [], 'rf': []}
best_params = {'ridge': [], 'rf': []}

# Hyperparameters
ridge_params = {'regressor__alpha': [0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100]}
rf_params = {
    'regressor__n_estimators': [50, 100],
    'regressor__max_depth': [None, 10, 20],
    'regressor__min_samples_split': [2, 5],
    'regressor__min_samples_leaf': [1, 2],
    'regressor__bootstrap': [True],
    'regressor__max_features': ['sqrt', 'log2']
}

def train_and_evaluate_model(model_pipeline, X_train, y_train, group_train, param_grid, inner_cv):
    grid_search = GridSearchCV(estimator=model_pipeline, param_grid=param_grid, cv=inner_cv, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train, groups=group_train)
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    return best_model, best_score, best_params

# Define the outer cross-validation method
outer_cv = GroupKFold(n_splits=5)

for fold, (train_index, test_index) in enumerate(outer_cv.split(X, y, groups), start=1):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    group_train, group_test = groups.iloc[train_index], groups.iloc[test_index]

    print(f"\nFold {fold}:")
    print(f"    Training groups: {groups.iloc[train_index].unique()}")
    print(f"    Testing groups: {groups.iloc[test_index].unique()}")

    inner_cv = GroupKFold(n_splits=5)

    # Baseline model: Predicting the mean
    mean_value = y_train.mean()
    baseline_pred = np.full_like(y_test, fill_value=mean_value, dtype=np.float64)
    baseline_mse = mean_squared_error(y_test, baseline_pred)
    baseline_rmse = np.sqrt(baseline_mse)
    baseline_mae = mean_absolute_error(y_test, baseline_pred)
    baseline_r2 = r2_score(y_test, baseline_pred)
    results['baseline'].append((baseline_mse, baseline_rmse, baseline_mae, baseline_r2))
    print(f"\n    Baseline model Test MSE: {baseline_mse:.4f}")

    # Ridge Regression
    lr_model = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', Ridge())
    ])
    lr_best_model, lr_best_score, lr_best_params = train_and_evaluate_model(lr_model, X_train, y_train, group_train, ridge_params, inner_cv)
    lr_best_pred = lr_best_model.predict(X_test)
    lr_best_mse = mean_squared_error(y_test, lr_best_pred)
    lr_best_rmse = np.sqrt(lr_best_mse)
    lr_best_mae = mean_absolute_error(y_test, lr_best_pred)
    lr_best_r2 = r2_score(y_test, lr_best_pred)
    results['ridge'].append((lr_best_mse, lr_best_rmse, lr_best_mae, lr_best_r2))
    best_params['ridge'].append(lr_best_params)
    print(f"\n    Ridge Regression best parameters: {lr_best_params}")
    print(f"    Ridge Regression Test MSE: {lr_best_mse:.4f}")

    # Random Forest Regressor
    rf_model = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(random_state=42))
    ])
    rf_best_model, rf_best_score, rf_best_params = train_and_evaluate_model(rf_model, X_train, y_train, group_train, rf_params, inner_cv)
    rf_best_pred = rf_best_model.predict(X_test)
    rf_best_mse = mean_squared_error(y_test, rf_best_pred)
    rf_best_rmse = np.sqrt(rf_best_mse)
    rf_best_mae = mean_absolute_error(y_test, rf_best_pred)
    rf_best_r2 = r2_score(y_test, rf_best_pred)
    results['rf'].append((rf_best_mse, rf_best_rmse, rf_best_mae, rf_best_r2))
    best_params['rf'].append(rf_best_params)
    print(f"\n    Random Forest best parameters: {rf_best_params}")
    print(f"    Random Forest Test MSE: {rf_best_mse:.4f}")

# Calculate aggregate metrics and save to file
def calculate_and_print_summary_stats(model_name, metrics):
    mean_mse = np.mean([m[0] for m in metrics])
    std_mse = np.std([m[0] for m in metrics], ddof=1)
    mean_rmse = np.mean([m[1] for m in metrics])
    std_rmse = np.std([m[1] for m in metrics], ddof=1)
    mean_mae = np.mean([m[2] for m in metrics])
    std_mae = np.std([m[2] for m in metrics], ddof=1)
    mean_r2 = np.mean([m[3] for m in metrics])
    std_r2 = np.std([m[3] for m in metrics], ddof=1)
    n = len(metrics)
    confidence = 0.95
    mse_h = std_mse * t.ppf((1 + confidence) / 2, n - 1) / (n ** 0.5)
    rmse_h = std_rmse * t.ppf((1 + confidence) / 2, n - 1) / (n ** 0.5)
    mae_h = std_mae * t.ppf((1 + confidence) / 2, n - 1) / (n ** 0.5)
    r2_h = std_r2 * t.ppf((1 + confidence) / 2, n - 1) / (n ** 0.5)
    ci_lower_mse = mean_mse - mse_h
    ci_upper_mse = mean_mse + mse_h
    ci_lower_rmse = mean_rmse - rmse_h
    ci_upper_rmse = mean_rmse + rmse_h
    ci_lower_mae = mean_mae - mae_h
    ci_upper_mae = mean_mae + mae_h
    ci_lower_r2 = mean_r2 - r2_h
    ci_upper_r2 = mean_r2 + r2_h

    print(f"\nSummary for {model_name}:")
    print(f"    Mean MSE: {mean_mse:.4f} (95% CI: {ci_lower_mse:.4f} - {ci_upper_mse:.4f})")
    print(f"    Mean RMSE: {mean_rmse:.4f} (95% CI: {ci_lower_rmse:.4f} - {ci_upper_rmse:.4f})")
    print(f"    Mean MAE: {mean_mae:.4f} (95% CI: {ci_lower_mae:.4f} - {ci_upper_mae:.4f})")
    print(f"    Mean R2: {mean_r2:.4f} (95% CI: {ci_lower_r2:.4f} - {ci_upper_r2:.4f})")

# Perform paired t-tests to compare models
ridge_mses = [m[0] for m in results['ridge']]
rf_mses = [m[0] for m in results['rf']]
baseline_mses = [m[0] for m in results['baseline']]
ridge_maes = [m[2] for m in results['ridge']]
rf_maes = [m[2] for m in results['rf']]
baseline_maes = [m[2] for m in results['baseline']]
ridge_r2s = [m[3] for m in results['ridge']]
rf_r2s = [m[3] for m in results['rf']]
baseline_r2s = [m[3] for m in results['baseline']]

t_stat_mse_ridge_rf, p_value_mse_ridge_rf = ttest_rel(ridge_mses, rf_mses)
t_stat_mae_ridge_rf, p_value_mae_ridge_rf = ttest_rel(ridge_maes, rf_maes)
t_stat_r2_ridge_rf, p_value_r2_ridge_rf = ttest_rel(ridge_r2s, rf_r2s)

t_stat_mse_ridge_baseline, p_value_mse_ridge_baseline = ttest_rel(ridge_mses, baseline_mses)
t_stat_mae_ridge_baseline, p_value_mae_ridge_baseline = ttest_rel(ridge_maes, baseline_maes)
t_stat_r2_ridge_baseline, p_value_r2_ridge_baseline = ttest_rel(ridge_r2s, baseline_r2s)

t_stat_mse_rf_baseline, p_value_mse_rf_baseline = ttest_rel(rf_mses, baseline_mses)
t_stat_mae_rf_baseline, p_value_mae_rf_baseline = ttest_rel(rf_maes, baseline_maes)
t_stat_r2_rf_baseline, p_value_r2_rf_baseline = ttest_rel(rf_r2s, baseline_r2s)

print(f"\nPaired t-test between Ridge and Random Forest MSEs:")
print(f"t-statistic: {t_stat_mse_ridge_rf:.4f}, p-value: {p_value_mse_ridge_rf:.4f}")
print(f"\nPaired t-test between Ridge and Random Forest MAEs:")
print(f"t-statistic: {t_stat_mae_ridge_rf:.4f}, p-value: {p_value_mae_ridge_rf:.4f}")
print(f"\nPaired t-test between Ridge and Random Forest R2s:")
print(f"t-statistic: {t_stat_r2_ridge_rf:.4f}, p-value: {p_value_r2_ridge_rf:.4f}")

print(f"\nPaired t-test between Ridge and Baseline MSEs:")
print(f"t-statistic: {t_stat_mse_ridge_baseline:.4f}, p-value: {p_value_mse_ridge_baseline:.4f}")
print(f"\nPaired t-test between Ridge and Baseline MAEs:")
print(f"t-statistic: {t_stat_mae_ridge_baseline:.4f}, p-value: {p_value_mae_ridge_baseline:.4f}")
print(f"\nPaired t-test between Ridge and Baseline R2s:")
print(f"t-statistic: {t_stat_r2_ridge_baseline:.4f}, p-value: {p_value_r2_ridge_baseline:.4f}")

print(f"\nPaired t-test between Random Forest and Baseline MSEs:")
print(f"t-statistic: {t_stat_mse_rf_baseline:.4f}, p-value: {p_value_mse_rf_baseline:.4f}")
print(f"\nPaired t-test between Random Forest and Baseline MAEs:")
print(f"t-statistic: {t_stat_mae_rf_baseline:.4f}, p-value: {p_value_mae_rf_baseline:.4f}")
print(f"\nPaired t-test between Random Forest and Baseline R2s:")
print(f"t-statistic: {t_stat_r2_rf_baseline:.4f}, p-value: {p_value_r2_rf_baseline:.4f}")

# Printing final results
print("\nModel Comparison Results:")
print(f"Ridge MSEs: {ridge_mses}")
print(f"Random Forest MSEs: {rf_mses}")
print(f"Baseline MSEs: {baseline_mses}")
print(f"Ridge MAEs: {ridge_maes}")
print(f"Random Forest MAEs: {rf_maes}")
print(f"Baseline MAEs: {baseline_maes}")
print(f"Ridge R2s: {ridge_r2s}")
print(f"Random Forest R2s: {rf_r2s}")
print(f"Baseline R2s: {baseline_r2s}")

print("\nStatistical test results for MSE:")
print(f"t-statistic: {t_stat_mse_ridge_rf}, p-value: {p_value_mse_ridge_rf}")
print(f"t-statistic: {t_stat_mse_ridge_baseline}, p-value: {p_value_mse_ridge_baseline}")
print(f"t-statistic: {t_stat_mse_rf_baseline}, p-value: {p_value_mse_rf_baseline}")

print("\nStatistical test results for MAE:")
print(f"t-statistic: {t_stat_mae_ridge_rf}, p-value: {p_value_mae_ridge_rf}")
print(f"t-statistic: {t_stat_mae_ridge_baseline}, p-value: {p_value_mae_ridge_baseline}")
print(f"t-statistic: {t_stat_mae_rf_baseline}, p-value: {p_value_mae_rf_baseline}")

print("\nStatistical test results for R2:")
print(f"t-statistic: {t_stat_r2_ridge_rf}, p-value: {p_value_r2_ridge_rf}")
print(f"t-statistic: {t_stat_r2_ridge_baseline}, p-value: {p_value_r2_ridge_baseline}")
print(f"t-statistic: {t_stat_r2_rf_baseline}, p-value: {p_value_r2_rf_baseline}")

# %%
