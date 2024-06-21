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
from scipy.stats import t, ttest_rel, chi2

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

# Hyperparameters
ridge_params = {'regressor__alpha': [0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100]}
rf_params = {
    'regressor__n_estimators': [50, 100],
    'regressor__max_depth': [None, 10, 20],
    'regressor__min_samples_split': [2, 5],
    'regressor__min_samples_leaf': [1, 2],
    'regressor__max_features': ['sqrt', 'log2']
}

def train_and_evaluate_model(model_pipeline, X_train, y_train, group_train, param_grid, inner_cv):
    grid_search = GridSearchCV(estimator=model_pipeline, param_grid=param_grid, cv=inner_cv, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train, groups=group_train)
    best_model = grid_search.best_estimator_
    best_score = grid_search.best_score_
    return best_model, best_score

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
    results['baseline'].append(baseline_mse)
    print(f"    Baseline model Test MSE: {baseline_mse:.4f}")

    # Ridge Regression
    lr_model = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', Ridge())
    ])
    lr_best_model, lr_best_score = train_and_evaluate_model(lr_model, X_train, y_train, group_train, ridge_params, inner_cv)
    lr_best_pred = lr_best_model.predict(X_test)
    lr_best_mse = mean_squared_error(y_test, lr_best_pred)
    results['ridge'].append(lr_best_mse)
    print(f"    Ridge Regression Test MSE: {lr_best_mse:.4f}")

    # Random Forest Regressor
    rf_model = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(random_state=42))
    ])
    rf_best_model, rf_best_score = train_and_evaluate_model(rf_model, X_train, y_train, group_train, rf_params, inner_cv)
    rf_best_pred = rf_best_model.predict(X_test)
    rf_best_mse = mean_squared_error(y_test, rf_best_pred)
    results['rf'].append(rf_best_mse)
    print(f"    Random Forest Test MSE: {rf_best_mse:.4f}")

# Calculate and print summary stats
def calculate_summary_stats(metrics):
    mean_val = np.mean(metrics)
    std_val = np.std(metrics, ddof=1)
    n = len(metrics)
    confidence = 0.95

    # Confidence interval for the mean using the t-distribution
    mean_h = std_val * t.ppf((1 + confidence) / 2, n - 1) / (n ** 0.5)

    # Confidence interval for the standard deviation using the chi-squared distribution
    alpha = 1 - confidence
    chi2_lower = chi2.ppf(alpha / 2, n - 1)
    chi2_upper = chi2.ppf(1 - alpha / 2, n - 1)
    std_lower = np.sqrt((n - 1) * std_val ** 2 / chi2_upper)
    std_upper = np.sqrt((n - 1) * std_val ** 2 / chi2_lower)
    std_h = (std_upper - std_lower) / 2  # Half-width of the confidence interval

    return mean_val, mean_h, std_val, std_h

for model_name, mses in results.items():
    mean_mse, mean_h, std_mse, std_h = calculate_summary_stats(mses)
    print(f"\nSummary for {model_name.capitalize()}:")
    print(f"    Mean MSE: {mean_mse:.4f} ± {mean_h:.4f}")
    print(f"    Std MSE: {std_mse:.4f} ± {std_h:.4f}")

# Perform paired t-tests to compare models
def perform_t_tests(model1, model2, name1, name2):
    t_stat, p_value = ttest_rel(results[model1], results[model2])
    print(f"\nPaired t-test between {name1} and {name2} on MSE:")
    print(f"    t-statistic: {t_stat:.4f}, p-value: {p_value:.4f}")

perform_t_tests('baseline', 'ridge', 'Baseline', 'Ridge')
perform_t_tests('baseline', 'rf', 'Baseline', 'Random Forest')
perform_t_tests('ridge', 'rf', 'Ridge', 'Random Forest')

# %%
