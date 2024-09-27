import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Create the dataset
data = {
    'EmployeeID': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'FirstName': ['John', 'Jane', 'Michael', 'Emily', 'David', 'Alice', 'Robert', 'Laura', 'James', 'Emma'],
    'LastName': ['Doe', 'Smith', 'Johnson', 'Williams', 'Brown', 'Davis', 'Wilson', 'Moore', 'Taylor', 'Anderson'],
    'Role': ['Software Engineer', 'Data Scientist', 'UX Designer', 'Product Manager', 'QA Engineer', 'DevOps Engineer', 'Backend Developer', 'Frontend Developer', 'HR Manager', 'Marketing Specialist'],
    'Department': ['Development', 'Data Science', 'Design', 'Management', 'Quality Assurance', 'Operations', 'Development', 'Development', 'Human Resources', 'Marketing'],
    'Salary': [70000, 75000, 68000, 80000, 65000, 72000, 71000, 69000, 73000, 67000],
    'StartDate': ['2022-01-15', '2021-06-01', '2023-03-10', '2020-11-01', '2022-07-15', '2021-09-20', '2023-02-05', '2022-10-25', '2020-12-01', '2021-08-15'],
    'Project': ['Project Alpha', 'Project Beta', 'Project Gamma', 'Project Delta', 'Project Epsilon', 'Project Zeta', 'Project Alpha', 'Project Beta', 'NA', 'Project Gamma'],
    'PerformanceRating': [4.5, 4.7, 4.2, 4.6, 4.1, 4.3, 4.4, 4.0, 4.8, 4.2]
}
df = pd.DataFrame(data)
df['StartDate'] = pd.to_datetime(df['StartDate'])

# Easy Tasks
## 1. Create a Bar Plot
plt.figure(figsize=(10, 6))
sns.barplot(x='Department', y='Salary', data=df, estimator='mean')
plt.title('Average Salary by Department')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

## 2. Count Employees by Department
plt.figure(figsize=(10, 6))
sns.countplot(x='Department', data=df)
plt.title('Number of Employees by Department')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

## 3. Display Salary Distribution
plt.figure(figsize=(10, 6))
sns.histplot(df['Salary'], kde=True)
plt.title('Salary Distribution')
plt.tight_layout()
plt.show()

# Middle Tasks
## 4. Scatter Plot: Salary vs Performance Rating
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Salary', y='PerformanceRating', hue='Department', data=df, palette='tab10')
plt.title('Salary vs Performance Rating')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

## 5. Box Plot of Salary Distribution by Department
plt.figure(figsize=(10, 6))
sns.boxplot(x='Department', y='Salary', data=df)
plt.title('Salary Distribution by Department')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

## 6. Line Plot: Salary Over Time
plt.figure(figsize=(10, 6))
sns.lineplot(x='StartDate', y='Salary', data=df, marker='o')
plt.title('Salary Over Time')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Hard Tasks
## 7. Heatmap of Correlation Matrix
plt.figure(figsize=(10, 6))
sns.heatmap(df[['Salary', 'PerformanceRating']].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix Heatmap')
plt.tight_layout()
plt.show()

## 8. Violin Plot of Salary by Department
plt.figure(figsize=(10, 6))
sns.violinplot(x='Department', y='Salary', data=df)
plt.title('Violin Plot of Salary by Department')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

## 9. Pair Plot for Numerical Features
plt.figure(figsize=(10, 6))
sns.pairplot(df[['Salary', 'PerformanceRating']])
plt.title('Pair Plot for Numerical Features')
plt.tight_layout()
plt.show()

# Extreme Tasks
## 10. FacetGrid: Salary vs Performance for Each Project
g = sns.FacetGrid(df, col="Project", col_wrap=3, height=4)
g.map(sns.scatterplot, 'Salary', 'PerformanceRating')
plt.show()

## 11. Create Custom Aesthetic for Strip Plot
plt.figure(figsize=(10, 6))
sns.set_style("whitegrid")
sns.stripplot(x='Role', y='PerformanceRating', data=df, jitter=True, palette='Set1')
plt.title('Performance Rating by Role')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

## 12. Combine Multiple Plots in a Grid
fig, axs = plt.subplots(2, 2, figsize=(15, 12))
sns.scatterplot(x='Salary', y='PerformanceRating', hue='Department', data=df, ax=axs[0, 0], palette='tab10').set_title('Scatter Plot')
sns.barplot(x='Department', y='Salary', data=df, estimator='mean', ax=axs[0, 1]).set_title('Bar Plot')
sns.countplot(x='Department', data=df, ax=axs[1, 0]).set_title('Count Plot')
sns.histplot(df['Salary'], kde=True, ax=axs[1, 1]).set_title('Salary Distribution')
plt.tight_layout()
plt.show()
