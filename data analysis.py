#%%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
file_path = 'HR_data.csv'
data = pd.read_csv(file_path)

# Select the HR features and the target variable
hr_features = ['HR_Mean', 'HR_Median', 'HR_std', 'HR_Min', 'HR_Max', 'HR_AUC']
all_features = hr_features + ['Frustrated']
group = 'Individual'

# Calculate within-group correlations and aggregate them
within_group_corr = data.groupby(group).apply(lambda x: x[all_features].corr())
mean_within_group_corr = within_group_corr.groupby(level=1).mean()

# Calculate between-group correlations
grouped_data = data.groupby(group).mean().reset_index()
between_group_corr = grouped_data[all_features].corr()

# Ensure consistent order of columns and indices
ordered_columns = all_features
mean_within_group_corr = mean_within_group_corr.loc[ordered_columns, ordered_columns]
between_group_corr = between_group_corr.loc[ordered_columns, ordered_columns]

# Plot mean within-group correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(mean_within_group_corr, annot=True, cmap='coolwarm', fmt='.2f', cbar_kws={'label': 'Correlation Coefficient'})
plt.title('Mean Within-Group Correlation Matrix', fontsize=16)
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.show()

# Create a pair plot with grouping
plt.figure(figsize=(12, 10))
sns.pairplot(data, vars=all_features, hue=group, diag_kind='kde', palette='tab10')
plt.suptitle('Grouped Pair Plot of HR Features and Frustration Level', y=1.02)
plt.show()

# %%
