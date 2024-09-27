import pandas as pd
import matplotlib.pyplot as plt

# Assuming you have a DataFrame called 'df' with the necessary columns
# If not, you'll need to load your data first:
df = pd.read_csv('startup_data.csv')

# Set the style for all plots
plt.style.use('ggplot')

df.columns = df.columns.str.strip()

# Task 1: Bar Plot - Average Salary by Department
plt.figure(figsize=(12, 6))
avg_salary = df.groupby('Department')['Salary'].mean().sort_values(ascending=False)
avg_salary.plot(kind='bar')
plt.title('Average Salary by Department')
plt.xlabel('Department')
plt.ylabel('Average Salary')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Task 2: Scatter Plot - Salary vs Performance Rating
plt.figure(figsize=(10, 6))
plt.scatter(df['PerformanceRating'], df['Salary'])
plt.title('Salary vs Performance Rating')
plt.xlabel('Performance Rating')
plt.ylabel('Salary')
plt.tight_layout()
plt.show()

# Task 3: Box Plot - Salary Distribution by Department
plt.figure(figsize=(12, 6))
df.boxplot(column='Salary', by='Department')
plt.title('Salary Distribution by Department')
plt.suptitle('')  # Remove automatic suptitle
plt.xlabel('Department')
plt.ylabel('Salary')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Task 4: Count Plot - Number of Employees by Department
plt.figure(figsize=(10, 6))
df['Department'].value_counts().plot(kind='bar')
plt.title('Number of Employees by Department')
plt.xlabel('Department')
plt.ylabel('Number of Employees')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Task 5: Line Plot - Salary Over Time (Start Dates)
df['StartDate'] = pd.to_datetime(df['StartDate'])
df_sorted = df.sort_values('StartDate')
plt.figure(figsize=(12, 6))
plt.plot(df_sorted['StartDate'], df_sorted['Salary'])
plt.title('Salary Over Time (Start Dates)')
plt.xlabel('Start Date')
plt.ylabel('Salary')
plt.tight_layout()
plt.show()

# Task 6: Heatmap - Correlation Matrix
plt.figure(figsize=(10, 8))
corr_matrix = df.select_dtypes(include=['float64', 'int64']).corr()
plt.imshow(corr_matrix, cmap='coolwarm')
plt.colorbar()
plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=90)
plt.yticks(range(len(corr_matrix.columns)), corr_matrix.columns)
plt.title('Correlation Matrix Heatmap')
plt.tight_layout()
plt.show()

# Task 7: Violin Plot - Salary Distribution by Department
plt.figure(figsize=(12, 6))
departments = df['Department'].unique()
violins = [df[df['Department'] == dept]['Salary'] for dept in departments]
plt.violinplot(violins, showmeans=True, showmedians=True)
plt.xticks(range(1, len(departments) + 1), departments, rotation=45)
plt.title('Salary Distribution by Department (Violin Plot)')
plt.xlabel('Department')
plt.ylabel('Salary')
plt.tight_layout()
plt.show()

# Task 8: Pair Plot - Relationships Between Numerical Features
numerical_features = df.select_dtypes(include=['float64', 'int64']).columns
fig, axes = plt.subplots(len(numerical_features), len(numerical_features), figsize=(20, 20))
for i, feature1 in enumerate(numerical_features):
    for j, feature2 in enumerate(numerical_features):
        if i != j:
            axes[i, j].scatter(df[feature2], df[feature1], alpha=0.5)
        else:
            axes[i, j].hist(df[feature1], bins=30)
        if i == len(numerical_features) - 1:
            axes[i, j].set_xlabel(feature2)
        if j == 0:
            axes[i, j].set_ylabel(feature1)
plt.tight_layout()
plt.show()

# Task 9: Histogram - Salary Distribution
plt.figure(figsize=(10, 6))
plt.hist(df['Salary'], bins=30, edgecolor='black')
plt.title('Salary Distribution')
plt.xlabel('Salary')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# Task 10: SubPlot - Performance Rating by Project
projects = df['Project'].unique()
num_projects = len(projects)

# Calculate grid size based on the number of unique projects
rows = (num_projects // 2) + (num_projects % 2)
cols = 2 if num_projects > 1 else 1

fig, axes = plt.subplots(rows, cols, figsize=(15, 7 * rows))

# Flatten axes only if there are multiple plots
if num_projects > 1:
    axes = axes.ravel()

# Plot each project
for i, project in enumerate(projects):
    project_data = df[df['Project'] == project]
    axes[i].scatter(project_data['PerformanceRating'], project_data['Salary'])
    axes[i].set_title(f'Project: {project}')
    axes[i].set_xlabel('Performance Rating')
    axes[i].set_ylabel('Salary')

# Remove any empty axes if the number of projects is less than the grid size
if num_projects < len(axes):
    for j in range(num_projects, len(axes)):
        fig.delaxes(axes[j])

plt.suptitle('Salary vs Performance Rating by Project', y=1.02)
plt.tight_layout()
plt.show()

# Task 11: Strip Plot - Performance Rating by Role
plt.figure(figsize=(12, 6))
roles = df['Role'].unique()
for i, role in enumerate(roles):
    role_data = df[df['Role'] == role]['PerformanceRating']
    plt.scatter([i] * len(role_data), role_data, alpha=0.5)
plt.xticks(range(len(roles)), roles, rotation=45)
plt.title('Performance Rating by Role')
plt.xlabel('Role')
plt.ylabel('Performance Rating')
plt.tight_layout()
plt.show()