import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('startup_data.csv')
df.columns = df.columns.str.strip()

df['StartDate'] = pd.to_datetime(df['StartDate'])

# Set the style for all plots
sns.set_style("whitegrid")

# Task 1: Bar Plot - Average Salary by Department
plt.figure(figsize=(12, 6))
sns.barplot(x='Department', y='Salary', data=df)
plt.title('Average Salary by Department')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('task1_bar_plot_salary_by_department.png')
plt.close()

# Task 2: Scatter Plot - Salary vs Performance Rating
plt.figure(figsize=(10, 6))
sns.scatterplot(x='PerformanceRating', y='Salary', data=df)
plt.title('Salary vs Performance Rating')
plt.tight_layout()
plt.savefig('task2_scatter_plot_salary_vs_performance.png')
plt.close()

# Task 3: Box Plot - Salary Distribution by Department
plt.figure(figsize=(12, 6))
sns.boxplot(x='Department', y='Salary', data=df)
plt.title('Salary Distribution by Department')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('task3_box_plot_salary_distribution.png')
plt.close()

# Task 4: Count Plot - Number of Employees by Department
plt.figure(figsize=(10, 6))
sns.countplot(x='Department', data=df)
plt.title('Number of Employees by Department')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('task4_count_plot_employees_by_department.png')
plt.close()

# Task 5: Line Plot - Salary Over Time (Start Dates)
plt.figure(figsize=(12, 6))
sns.lineplot(x='StartDate', y='Salary', data=df)
plt.title('Salary Over Time (Start Dates)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('task5_line_plot_salary_over_time.png')
plt.close()

# Task 6: Heatmap - Correlation Matrix
plt.figure(figsize=(12, 10))
correlation_matrix = df.select_dtypes(include=['float64', 'int64']).corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.tight_layout()
plt.savefig('task6_heatmap_correlation_matrix.png')
plt.close()

# Task 7: Violin Plot - Salary Distribution by Department
plt.figure(figsize=(12, 6))
sns.violinplot(x='Department', y='Salary', data=df)
plt.title('Salary Distribution by Department')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('task7_violin_plot_salary_distribution.png')
plt.close()

# Task 8: Pair Plot - Relationships Between Numerical Features
numerical_features = ['Salary', 'PerformanceRating', 'YearsOfExperience', 'Age']
sns.pairplot(df[numerical_features])
plt.tight_layout()
plt.savefig('task8_pair_plot_numerical_features.png')
plt.close()

# Task 9: Histogram - Salary Distribution
plt.figure(figsize=(10, 6))
sns.histplot(df['Salary'], kde=True)
plt.title('Salary Distribution')
plt.tight_layout()
plt.savefig('task9_histogram_salary_distribution.png')
plt.close()

# Task 10: FacetGrid - Performance Rating by Project
g = sns.FacetGrid(df, col="Project", col_wrap=3, height=4, aspect=1.5)
g.map(sns.scatterplot, "PerformanceRating", "Salary")
g.add_legend()
g.fig.suptitle('Performance Rating vs Salary by Project', y=1.02)
plt.tight_layout()
plt.savefig('task10_facetgrid_performance_by_project.png')
plt.close()

# Task 11: Strip Plot - Performance Rating by Role
plt.figure(figsize=(12, 6))
sns.stripplot(x='Role', y='PerformanceRating', data=df, jitter=True)
plt.title('Performance Rating by Role')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('task11_strip_plot_performance_by_role.png')
plt.close()

print("All visualizations have been saved as PNG files in the current directory.")